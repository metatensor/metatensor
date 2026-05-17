//! Selective loading of `TensorMap` from `.mts` files.
//!
//! `load_partial` accepts three optional `Labels`-shaped filters
//! (`keys`, `samples`, `properties`); the loader memory-maps the file to
//! find block boundaries, then byte-copies only the selected rows /
//! columns into freshly-allocated arrays returned by the standard
//! `create_array` callback. The result of `load_partial(path, None,
//! None, None, cb)` equals `load(path, cb)`.

use std::collections::HashMap;
use std::ffi::CString;
use std::io::Cursor;
use std::sync::Arc;

use dlpk::sys::{DLDataType, DLDevice, DLPackVersion};
use memmap2::Mmap;
use zip::ZipArchive;

use super::labels::load_labels;
use super::{npy_layout, parse_stored_npy_entry};

use crate::labels::LabelValue;
use crate::utils::ConstCString;
use crate::{mts_array_t, Error, Labels, TensorBlock, TensorMap};


/// Load a `TensorMap` from the file at the given path, selecting only a
/// subset of the data based on `keys`, `samples`, and `properties`.
///
/// The file is memory-mapped for fast random access; only the selected
/// rows and columns are copied into arrays allocated by `create_array`.
/// The returned tensor owns its data (no live mmap reference); the
/// underlying file is unmapped before this function returns.
///
/// Each selection uses `Labels::select` semantics: the selection's
/// dimensions must be a subset of the target's dimensions, and all
/// matching rows are kept. `None` for any of `keys` / `samples` /
/// `properties` means "select all" on that dimension.
///
/// `create_array` follows the same contract as `mts_create_array_callback_t`:
/// it gets `(shape, dtype)` and must return an `mts_array_t` of that
/// shape and dtype.
///
/// # File format constraints
/// - The file must use the `STORED` (uncompressed) ZIP format that
///   `mts_*_save` writes.
/// - Numeric arrays must use native byte order.
pub fn load_partial<F>(
    path: &str,
    keys: Option<&Labels>,
    samples: Option<&Labels>,
    properties: Option<&Labels>,
    create_array: F,
) -> Result<TensorMap, Error>
where
    F: Fn(Vec<usize>, DLDataType) -> Result<mts_array_t, Error>,
{
    let file = std::fs::File::open(path)?;
    // SAFETY: read-only view; file owned for the duration of this call.
    let mmap = unsafe { Mmap::map(&file) }.map_err(Error::Io)?;

    let cursor = Cursor::new(mmap.as_ref());
    let mut archive = ZipArchive::new(cursor).map_err(|e| ("<root>".into(), e))?;

    let keys_path = String::from("keys.npy");
    let all_keys = load_labels(archive.by_name(&keys_path).map_err(|e| (keys_path, e))?)?;

    let block_indices = compute_indices(&all_keys, keys)?;

    let filtered_keys = labels_subset(&all_keys, &block_indices)?;

    let mut blocks = Vec::with_capacity(block_indices.len());
    for &block_i in &block_indices {
        let prefix = format!("blocks/{}/", block_i);
        let block = read_partial_block(
            &mut archive,
            mmap.as_ref(),
            &prefix,
            samples,
            properties,
            None,
            &create_array,
        )?;
        blocks.push(block);
    }

    let mut tensor = TensorMap::new(Arc::new(filtered_keys), blocks)?;

    super::load_info_json(&mut archive, |key, value| {
        tensor.add_info(
            key,
            ConstCString::new(
                CString::new(value).expect("value in 'info.json' should not contain a NUL byte"),
            ),
        );
    })?;

    Ok(tensor)
}



/// Load a single `TensorBlock` from the file at the given path, selecting
/// only a subset of samples and properties. See [`load_partial`] for the
/// filter semantics and file-format constraints.
pub fn load_block_partial<F>(
    path: &str,
    samples: Option<&Labels>,
    properties: Option<&Labels>,
    create_array: F,
) -> Result<TensorBlock, Error>
where
    F: Fn(Vec<usize>, DLDataType) -> Result<mts_array_t, Error>,
{
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(Error::Io)?;

    let cursor = Cursor::new(mmap.as_ref());
    let mut archive = ZipArchive::new(cursor).map_err(|e| ("<root>".into(), e))?;

    read_partial_block(
        &mut archive,
        mmap.as_ref(),
        "",
        samples,
        properties,
        None,
        &create_array,
    )
}


/// Build a new `Labels` containing only the rows at the given indices.
/// Indices come from `select()` or sequential iteration, so uniqueness
/// is preserved.
fn labels_subset(labels: &Labels, indices: &[usize]) -> Result<Labels, Error> {
    let dimensions = labels.dimensions();
    if dimensions.is_empty() {
        return Labels::from_vec(&dimensions, Vec::new());
    }
    let size = dimensions.len();
    let mut values = Vec::with_capacity(indices.len() * size);
    let cpu_labels = labels.to_cpu();
    for &idx in indices {
        values.extend_from_slice(&cpu_labels[idx]);
    }
    // SAFETY: indices come from select() or 0..count(), so the resulting
    // subset has the same uniqueness guarantee as the source labels.
    unsafe { Labels::from_vec_unchecked_uniqueness(&dimensions, values) }
}


/// Compute the index list for a sample / property selector, materialising
/// `0..count()` when the selector is `None` and otherwise running
/// `Labels::select`.
fn compute_indices(all: &Labels, selector: Option<&Labels>) -> Result<Vec<usize>, Error> {
    if let Some(sel) = selector {
        let mut selected = vec![0u64; all.count()];
        let n = all.select(sel, &mut selected)?;
        selected[..n]
            .iter()
            .map(|&i| {
                usize::try_from(i).map_err(|_| {
                    Error::Internal(format!("label index {} overflows usize", i))
                })
            })
            .collect::<Result<Vec<_>, _>>()
    } else {
        Ok((0..all.count()).collect())
    }
}


/// Filter gradient samples whose column-0 (parent-sample index) is in
/// the kept parent set, and rewrite column-0 with the new sequential
/// index.
fn reindex_gradient_samples(
    grad_samples: &Labels,
    parent_sample_indices: &[usize],
) -> Result<(Vec<usize>, Arc<Labels>), Error> {
    let mut parent_map: HashMap<i32, i32> = HashMap::new();
    for (new_idx, &old_idx) in parent_sample_indices.iter().enumerate() {
        parent_map.insert(old_idx as i32, new_idx as i32);
    }

    let grad_dimensions = grad_samples.dimensions();
    let mut kept_grad_indices = Vec::new();
    let mut new_grad_values: Vec<LabelValue> = Vec::new();
    let cpu_grad_samples = grad_samples.to_cpu();

    for (i, entry) in cpu_grad_samples.iter().enumerate() {
        let parent_sample_idx = entry[0];
        if let Some(&new_parent_idx) = parent_map.get(&parent_sample_idx) {
            kept_grad_indices.push(i);
            new_grad_values.push(LabelValue::from(new_parent_idx));
            for &val in &entry[1..] {
                new_grad_values.push(val);
            }
        }
    }

    // SAFETY: the source labels are unique; filtering rows + rewriting
    // column-0 via a 1:1 map preserves uniqueness.
    let new_grad_samples = Arc::new(unsafe {
        Labels::from_vec_unchecked_uniqueness(&grad_dimensions, new_grad_values)?
    });
    Ok((kept_grad_indices, new_grad_samples))
}


/// Load the `components/{i}.npy` labels for a block (one per non-edge
/// shape dimension).
fn load_components(
    archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
    prefix: &str,
    shape_len: usize,
) -> Result<Vec<Arc<Labels>>, Error> {
    let mut components = Vec::new();
    for i in 0..shape_len.saturating_sub(2) {
        let comp_path = format!("{}components/{}.npy", prefix, i);
        let comp_file = archive.by_name(&comp_path).map_err(|e| (comp_path, e))?;
        components.push(Arc::new(load_labels(comp_file)?));
    }
    Ok(components)
}


/// Group a sorted-or-unsorted index list into maximal contiguous runs.
/// Returns a list of `(new_col_start, old_col_start, run_len)` triples
/// such that for `r in 0..run_len`, `prop_idx[new_col_start + r] ==
/// old_col_start + r`.
fn contiguous_runs(prop_idx: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut runs = Vec::new();
    if prop_idx.is_empty() {
        return runs;
    }
    let mut new_col_start = 0;
    let mut old_col_start = prop_idx[0];
    let mut run_len = 1;
    for (i, &col) in prop_idx.iter().enumerate().skip(1) {
        if col == old_col_start + run_len {
            run_len += 1;
        } else {
            runs.push((new_col_start, old_col_start, run_len));
            new_col_start = i;
            old_col_start = col;
            run_len = 1;
        }
    }
    runs.push((new_col_start, old_col_start, run_len));
    runs
}

fn row_major_strides(shape: &[usize]) -> Result<Vec<i64>, Error> {
    let mut strides = vec![1; shape.len()];
    let mut next = 1i64;
    for dim in (0..shape.len()).rev() {
        strides[dim] = next;
        let size = i64::try_from(shape[dim]).map_err(|_| {
            Error::Serialization(format!("array dimension {} does not fit in i64", shape[dim]))
        })?;
        next = next.checked_mul(size).ok_or_else(|| {
            Error::Serialization(format!(
                "array shape {:?} has overflowing contiguous strides",
                shape
            ))
        })?;
    }
    Ok(strides)
}

fn dlpack_strides(shape: &[usize], strides: Option<&[i64]>) -> Result<Vec<i64>, Error> {
    let strides = match strides {
        Some(strides) => {
            if strides.len() != shape.len() {
                return Err(Error::Serialization(format!(
                    "create_array returned strides {:?} for shape {:?}",
                    strides, shape
                )));
            }
            strides.to_vec()
        }
        None => row_major_strides(shape)?,
    };

    for (&dim, &stride) in shape.iter().zip(&strides) {
        if stride < 0 {
            return Err(Error::Serialization(format!(
                "create_array returned negative stride {} for shape {:?}",
                stride, shape
            )));
        }
        if dim > 1 && stride == 0 {
            return Err(Error::Serialization(format!(
                "create_array returned zero stride for non-singleton shape {:?}",
                shape
            )));
        }
    }

    Ok(strides)
}

fn is_row_major_contiguous(shape: &[usize], strides: &[i64]) -> Result<bool, Error> {
    if shape.len() != strides.len() {
        return Ok(false);
    }

    let expected = row_major_strides(shape)?;
    Ok(shape
        .iter()
        .zip(strides)
        .zip(expected)
        .all(|((&dim, &stride), expected)| dim == 1 || stride == expected))
}

fn component_offsets(shape: &[usize], strides: &[i64]) -> Result<Vec<i64>, Error> {
    let n_components = shape.iter().product();
    let mut offsets = Vec::with_capacity(n_components);

    for mut flat in 0..n_components {
        let mut offset = 0i64;
        for (&size, &stride) in shape.iter().zip(strides).rev() {
            let index = flat % size;
            flat /= size;
            let index = i64::try_from(index).map_err(|_| {
                Error::Serialization(format!("component index {} does not fit in i64", index))
            })?;
            offset = offset
                .checked_add(index.checked_mul(stride).ok_or_else(|| {
                    Error::Serialization("component stride offset overflowed".into())
                })?)
                .ok_or_else(|| {
                    Error::Serialization("component stride offset overflowed".into())
                })?;
        }
        offsets.push(offset);
    }

    Ok(offsets)
}

fn checked_dst_offset(
    row: usize,
    component_offset: i64,
    property: usize,
    strides: &[i64],
    elem_size: usize,
) -> Result<usize, Error> {
    let row = i64::try_from(row)
        .map_err(|_| Error::Serialization(format!("row index {} does not fit in i64", row)))?;
    let property = i64::try_from(property).map_err(|_| {
        Error::Serialization(format!("property index {} does not fit in i64", property))
    })?;

    let offset = row
        .checked_mul(strides[0])
        .and_then(|offset| offset.checked_add(component_offset))
        .and_then(|offset| offset.checked_add(property.checked_mul(*strides.last()?)?))
        .ok_or_else(|| Error::Serialization("destination stride offset overflowed".into()))?;

    let offset = usize::try_from(offset).map_err(|_| {
        Error::Serialization(format!("destination stride offset {} is negative", offset))
    })?;

    offset
        .checked_mul(elem_size)
        .ok_or_else(|| Error::Serialization("destination byte offset overflowed".into()))
}

fn validate_selected_indices(
    src_shape: &[usize],
    sample_indices: &[usize],
    prop_indices: Option<&[usize]>,
) -> Result<(), Error> {
    for &sample in sample_indices {
        if sample >= src_shape[0] {
            return Err(Error::Serialization(format!(
                "sample index {} is out of bounds for source shape {:?}",
                sample, src_shape
            )));
        }
    }

    if let Some(prop_indices) = prop_indices {
        let n_props = *src_shape.last().expect("block values must have rank >= 2");
        for &property in prop_indices {
            if property >= n_props {
                return Err(Error::Serialization(format!(
                    "property index {} is out of bounds for source shape {:?}",
                    property, src_shape
                )));
            }
        }
    }

    Ok(())
}


/// Copy the selected rows / columns from `mmap` (at `raw_data_offset`)
/// into a freshly-allocated array. The destination dtype must match the
/// source dtype's element width (the callback is given `dtype` so it
/// can allocate the right type).
fn gather_selected_data<F>(
    mmap: &[u8],
    raw_data_offset: usize,
    dtype: DLDataType,
    src_shape: &[usize],
    sample_indices: &[usize],
    prop_indices: Option<&[usize]>,
    create_array: &F,
) -> Result<mts_array_t, Error>
where
    F: Fn(Vec<usize>, DLDataType) -> Result<mts_array_t, Error>,
{
    if src_shape.len() < 2 {
        return Err(Error::Serialization(format!(
            "block values must have rank >= 2, got shape {:?}",
            src_shape
        )));
    }
    validate_selected_indices(src_shape, sample_indices, prop_indices)?;

    let (elem_size, n_components, src_n_props, src_row_bytes) = npy_layout(src_shape, dtype);

    let new_n_samples = sample_indices.len();
    let new_n_props = prop_indices.map_or(src_n_props, |p| p.len());

    let mut output_shape = Vec::with_capacity(src_shape.len());
    output_shape.push(new_n_samples);
    if src_shape.len() > 2 {
        output_shape.extend_from_slice(&src_shape[1..src_shape.len() - 1]);
    }
    output_shape.push(new_n_props);

    let total_elems: usize = output_shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .ok_or_else(|| {
            Error::Serialization(format!("output shape {:?} is too large", output_shape))
        })?;
    let output = create_array(output_shape.clone(), dtype)?;

    if new_n_samples == 0 || new_n_props == 0 || elem_size == 0 {
        return Ok(output);
    }

    let device = DLDevice::cpu();
    let version = DLPackVersion::current();
    let mut dl_tensor = output.as_dlpack(device, None, version)?;
    let tensor_ref = dl_tensor.as_mut();

    let dst_dtype = tensor_ref.dtype();
    let dst_elem_bytes = (dst_dtype.bits as usize / 8) * dst_dtype.lanes as usize;
    if dst_elem_bytes != elem_size {
        return Err(Error::Serialization(format!(
            "create_array returned an array with dtype bits={} lanes={}, \
             but the source NPY entry has bits={} lanes={} -- the callback \
             must return an array matching the file's dtype",
            dst_dtype.bits, dst_dtype.lanes, dtype.bits, dtype.lanes
        )));
    }

    if tensor_ref.device().device_type != DLDevice::cpu().device_type {
        return Err(Error::Serialization(
            "create_array returned a non-CPU DLPack tensor".into(),
        ));
    }

    let dst_shape = tensor_ref
        .shape()
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| {
                Error::Serialization(format!(
                    "create_array returned a negative or overflowing shape dimension {}",
                    dim
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if dst_shape != output_shape {
        return Err(Error::Serialization(format!(
            "create_array returned shape {:?}, expected {:?}",
            dst_shape, output_shape
        )));
    }

    let dst_strides = dlpack_strides(&output_shape, tensor_ref.strides())?;

    let total_bytes = total_elems
        .checked_mul(elem_size)
        .ok_or_else(|| Error::Serialization("destination byte length overflowed".into()))?;
    let dst_base = tensor_ref.as_dltensor().data as *mut u8;
    if dst_base.is_null() && total_bytes > 0 {
        return Err(Error::Serialization(
            "create_array returned an array with a NULL data pointer".into(),
        ));
    }
    let dst_ptr = dst_base.wrapping_add(tensor_ref.byte_offset());

    if !is_row_major_contiguous(&output_shape, &dst_strides)? {
        let component_shape = &src_shape[1..src_shape.len() - 1];
        let component_strides = &dst_strides[1..dst_strides.len() - 1];
        let component_offsets = component_offsets(component_shape, component_strides)?;
        let all_properties;
        let selected_properties = match prop_indices {
            Some(indices) => indices,
            None => {
                all_properties = (0..src_n_props).collect::<Vec<_>>();
                &all_properties
            }
        };

        for (new_row, &old_row) in sample_indices.iter().enumerate() {
            let src_offset = raw_data_offset + old_row * src_row_bytes;
            for (new_col, &old_col) in selected_properties.iter().enumerate() {
                for (comp, &comp_dst_offset) in component_offsets.iter().enumerate() {
                    let src_off = src_offset
                        + (comp * src_n_props + old_col)
                            .checked_mul(elem_size)
                            .ok_or_else(|| {
                                Error::Serialization("source byte offset overflowed".into())
                            })?;
                    let dst_off = checked_dst_offset(
                        new_row,
                        comp_dst_offset,
                        new_col,
                        &dst_strides,
                        elem_size,
                    )?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            mmap.as_ptr().add(src_off),
                            dst_ptr.add(dst_off),
                            elem_size,
                        );
                    }
                }
            }
        }

        return Ok(output);
    }

    // SAFETY: dst_elem_bytes == elem_size and the destination DLPack tensor is
    // contiguous after applying byte_offset. The slice borrows through
    // `dl_tensor`, which is dropped at the end of this function.
    let dst_bytes: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(dst_ptr, total_bytes) };

    if let Some(prop_idx) = prop_indices {
        let runs = contiguous_runs(prop_idx);
        let dst_row_bytes = n_components * new_n_props * elem_size;
        for (new_row, &old_row) in sample_indices.iter().enumerate() {
            let src_offset = raw_data_offset + old_row * src_row_bytes;
            let row_buf = &mmap[src_offset..src_offset + src_row_bytes];
            let dst_row_off = new_row * dst_row_bytes;
            for comp in 0..n_components {
                let src_comp_off = comp * src_n_props * elem_size;
                let dst_comp_off = dst_row_off + comp * new_n_props * elem_size;
                for &(new_col_start, old_col_start, run_len) in &runs {
                    let src_off = src_comp_off + old_col_start * elem_size;
                    let dst_off = dst_comp_off + new_col_start * elem_size;
                    let run_bytes = run_len * elem_size;
                    dst_bytes[dst_off..dst_off + run_bytes]
                        .copy_from_slice(&row_buf[src_off..src_off + run_bytes]);
                }
            }
        }
    } else {
        // Whole-row copy (no property filter).
        for (new_row, &old_row) in sample_indices.iter().enumerate() {
            let src_offset = raw_data_offset + old_row * src_row_bytes;
            let dst_offset = new_row * src_row_bytes;
            dst_bytes[dst_offset..dst_offset + src_row_bytes]
                .copy_from_slice(&mmap[src_offset..src_offset + src_row_bytes]);
        }
    }

    Ok(output)
}


/// Read one block with the given sample / property selectors. The
/// `parent_properties` parameter is set when recursing into gradient
/// blocks, where the gradient inherits its parent block's properties
/// (and the same property filter).
fn read_partial_block<F>(
    archive: &mut ZipArchive<Cursor<&[u8]>>,
    mmap: &[u8],
    prefix: &str,
    samples_sel: Option<&Labels>,
    properties_sel: Option<&Labels>,
    parent_properties: Option<(Arc<Labels>, Option<&[usize]>)>,
    create_array: &F,
) -> Result<TensorBlock, Error>
where
    F: Fn(Vec<usize>, DLDataType) -> Result<mts_array_t, Error>,
{
    let samples_path = format!("{}samples.npy", prefix);
    let samples_file = archive.by_name(&samples_path).map_err(|e| (samples_path, e))?;
    let all_samples = load_labels(samples_file)?;

    let values_path = format!("{}values.npy", prefix);
    let (src_shape, dtype, raw_data_offset) =
        parse_stored_npy_entry(archive, mmap, &values_path)?;

    let sample_indices = compute_indices(&all_samples, samples_sel)?;

    let (new_properties, prop_indices): (Arc<Labels>, Option<Vec<usize>>) =
        if let Some((parent_props, parent_prop_idx)) = parent_properties {
            (parent_props, parent_prop_idx.map(|p| p.to_vec()))
        } else {
            let props_path = format!("{}properties.npy", prefix);
            let props_file = archive.by_name(&props_path).map_err(|e| (props_path, e))?;
            let all_props = load_labels(props_file)?;
            if properties_sel.is_some() {
                let idx = compute_indices(&all_props, properties_sel)?;
                let filtered = labels_subset(&all_props, &idx)?;
                (Arc::new(filtered), Some(idx))
            } else {
                (Arc::new(all_props), None)
            }
        };

    let components = load_components(archive, prefix, src_shape.len())?;

    let data = gather_selected_data(
        mmap, raw_data_offset, dtype, &src_shape,
        &sample_indices, prop_indices.as_deref(), create_array,
    )?;

    let new_samples = Arc::new(labels_subset(&all_samples, &sample_indices)?);
    let mut block = TensorBlock::new(data, new_samples, components, new_properties.clone())?;

    for parameter in &super::discover_gradient_parameters(archive, prefix) {
        let grad_prefix = format!("{}gradients/{}/", prefix, parameter);
        let gradient = read_partial_gradient(
            archive,
            mmap,
            &grad_prefix,
            &sample_indices,
            &new_properties,
            prop_indices.as_deref(),
            create_array,
        )?;
        block.add_gradient(parameter, gradient)?;
    }

    Ok(block)
}


/// Read a gradient block with parent-sample reindexing.
///
/// Gradient samples have shape `[n_grad_samples, k+1]` where column 0
/// indexes into the parent block's samples. After we filter parent
/// samples, we (a) keep only gradient samples whose column-0 value is
/// in the kept set, (b) rewrite column 0 to the new sequential index.
fn read_partial_gradient<F>(
    archive: &mut ZipArchive<Cursor<&[u8]>>,
    mmap: &[u8],
    prefix: &str,
    parent_sample_indices: &[usize],
    new_properties: &Arc<Labels>,
    prop_indices: Option<&[usize]>,
    create_array: &F,
) -> Result<TensorBlock, Error>
where
    F: Fn(Vec<usize>, DLDataType) -> Result<mts_array_t, Error>,
{
    let samples_path = format!("{}samples.npy", prefix);
    let samples_file = archive.by_name(&samples_path).map_err(|e| (samples_path, e))?;
    let grad_samples = load_labels(samples_file)?;

    let values_path = format!("{}values.npy", prefix);
    let (src_shape, dtype, raw_data_offset) =
        parse_stored_npy_entry(archive, mmap, &values_path)?;

    let components = load_components(archive, prefix, src_shape.len())?;
    let (kept_grad_indices, new_grad_samples) =
        reindex_gradient_samples(&grad_samples, parent_sample_indices)?;

    let data = gather_selected_data(
        mmap, raw_data_offset, dtype, &src_shape,
        &kept_grad_indices, prop_indices, create_array,
    )?;

    let mut gradient = TensorBlock::new(data, new_grad_samples, components, new_properties.clone())?;

    for parameter in &super::discover_gradient_parameters(archive, prefix) {
        let grad_prefix = format!("{}gradients/{}/", prefix, parameter);
        let nested_gradient = read_partial_gradient(
            archive,
            mmap,
            &grad_prefix,
            &kept_grad_indices,
            new_properties,
            prop_indices,
            create_array,
        )?;
        gradient.add_gradient(parameter, nested_gradient)?;
    }

    Ok(gradient)
}

#[cfg(test)]
mod tests {
    use std::os::raw::c_void;

    use dlpk::sys::{
        DLDataType, DLDataTypeCode, DLManagedTensorVersioned, DLTensor, DLPackVersion,
    };

    use super::*;
    use crate::c_api::mts_status_t;

    struct StridedTestArray {
        shape_usize: Vec<usize>,
        shape_i64: Vec<i64>,
        strides: Vec<i64>,
        data: Vec<f64>,
        byte_offset: usize,
    }

    impl StridedTestArray {
        fn new(shape: Vec<usize>, strides: Vec<i64>, byte_offset: usize, len: usize) -> mts_array_t {
            let shape_i64 = shape.iter().map(|&dim| dim as i64).collect();
            let array = Box::new(StridedTestArray {
                shape_usize: shape,
                shape_i64,
                strides,
                data: vec![-1.0; len],
                byte_offset,
            });

            mts_array_t {
                ptr: Box::into_raw(array).cast(),
                origin: None,
                device: None,
                dtype: None,
                as_dlpack: Some(Self::as_dlpack),
                from_dlpack: None,
                shape: Some(Self::shape),
                reshape: None,
                swap_axes: None,
                create: None,
                copy: None,
                destroy: Some(Self::destroy),
                move_data: None,
            }
        }

        unsafe extern "C" fn as_dlpack(
            ptr: *mut c_void,
            dl_managed_tensor: *mut *mut DLManagedTensorVersioned,
            _device: DLDevice,
            _stream: *const i64,
            max_version: DLPackVersion,
        ) -> mts_status_t {
            let array = &mut *ptr.cast::<StridedTestArray>();
            let managed = Box::new(DLManagedTensorVersioned {
                version: max_version,
                manager_ctx: std::ptr::null_mut(),
                deleter: Some(Self::delete_dlpack),
                flags: 0,
                dl_tensor: DLTensor {
                    data: array.data.as_mut_ptr().cast(),
                    device: DLDevice::cpu(),
                    ndim: array.shape_i64.len() as i32,
                    dtype: DLDataType {
                        code: DLDataTypeCode::kDLFloat,
                        bits: 64,
                        lanes: 1,
                    },
                    shape: array.shape_i64.as_mut_ptr(),
                    strides: array.strides.as_mut_ptr(),
                    byte_offset: (array.byte_offset * std::mem::size_of::<f64>()) as u64,
                },
            });
            *dl_managed_tensor = Box::into_raw(managed);
            mts_status_t::MTS_SUCCESS
        }

        unsafe extern "C" fn shape(
            ptr: *const c_void,
            shape: *mut *const usize,
            shape_count: *mut usize,
        ) -> mts_status_t {
            let array = &*ptr.cast::<StridedTestArray>();
            *shape = array.shape_usize.as_ptr();
            *shape_count = array.shape_usize.len();
            mts_status_t::MTS_SUCCESS
        }

        unsafe extern "C" fn delete_dlpack(dlpack: *mut DLManagedTensorVersioned) {
            drop(Box::from_raw(dlpack));
        }

        unsafe extern "C" fn destroy(ptr: *mut c_void) {
            drop(Box::from_raw(ptr.cast::<StridedTestArray>()));
        }
    }

    #[test]
    fn gather_selected_data_respects_output_byte_offset_and_strides() {
        let raw_data_offset = 13;
        let mut mmap = vec![0; raw_data_offset];
        for value in 0..12 {
            mmap.extend_from_slice(&(value as f64).to_ne_bytes());
        }

        let dtype = DLDataType {
            code: DLDataTypeCode::kDLFloat,
            bits: 64,
            lanes: 1,
        };
        let output = gather_selected_data(
            &mmap,
            raw_data_offset,
            dtype,
            &[2, 2, 3],
            &[1],
            Some(&[2, 0]),
            &|shape, array_dtype| {
                assert_eq!(shape, vec![1, 2, 2]);
                assert_eq!(array_dtype, dtype);
                Ok(StridedTestArray::new(shape, vec![8, 3, 1], 1, 7))
            },
        )
        .unwrap();

        let array = unsafe { &*output.ptr.cast::<StridedTestArray>() };
        assert_eq!(array.data, vec![-1.0, 5.0, 3.0, -1.0, 8.0, 6.0, -1.0]);

        unsafe {
            output.destroy.unwrap()(output.ptr);
        }
    }
}
