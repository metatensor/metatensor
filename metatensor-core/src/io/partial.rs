use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::io::Read;
use std::sync::Arc;

use dlpk::sys::{DLDevice, DLPackVersion};
use memmap2::Mmap;

use super::labels::load_labels;
use super::Endianness;
use super::block::npy_descr_to_dtype;
use super::npy_header::{Header, DataType};

use crate::utils::ConstCString;
use crate::{TensorMap, TensorBlock, Labels, LabelValue, Error, mts_array_t};

/// Load a `TensorMap` from the file at the given path, selecting only a subset
/// of the data based on keys, samples, and properties.
///
/// This function memory-maps the file for efficient random access: only the
/// selected rows and columns are copied into the output arrays, avoiding
/// full-file reads. The file must use the STORED (uncompressed) ZIP format.
///
/// - `keys`: if `Some`, only blocks whose key matches the selection are loaded.
/// - `samples`: if `Some`, only rows matching the selection are kept.
/// - `properties`: if `Some`, only columns matching the selection are kept.
///
/// Each selection uses `Labels::select` semantics: the selection's names must
/// be a subset of the target's names, and all matching entries are included.
///
/// Arrays are created via the `create_array` callback (standard, not mmap).
/// The result is identical to calling `load()` and then filtering in memory.
pub fn load_partial<F>(
    path: &str,
    keys: Option<&Labels>,
    samples: Option<&Labels>,
    properties: Option<&Labels>,
    create_array: F,
) -> Result<TensorMap, Error>
where
    F: Fn(Vec<usize>) -> Result<mts_array_t, Error>
{
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(Error::Io)?;

    let cursor = std::io::Cursor::new(mmap.as_ref());
    let mut archive = zip::ZipArchive::new(cursor).map_err(|e| ("<root>".into(), e))?;

    // Load all keys
    let path_str = String::from("keys.npy");
    let all_keys = load_labels(archive.by_name(&path_str).map_err(|e| (path_str, e))?)?;

    // Filter by key selection
    let block_indices = if let Some(key_sel) = keys {
        let mut selected = vec![0i64; all_keys.count()];
        let n = all_keys.select(key_sel, &mut selected)?;
        selected[..n].iter().map(|&i| i as usize).collect::<Vec<_>>()
    } else {
        (0..all_keys.count()).collect()
    };

    let filtered_keys = labels_subset(&all_keys, &block_indices)?;

    let mut blocks = Vec::with_capacity(block_indices.len());
    for &block_i in &block_indices {
        let prefix = format!("blocks/{}/", block_i);
        let block = read_partial_block(
            &mut archive, &mmap, &prefix,
            samples, properties, None, &create_array,
        )?;
        blocks.push(block);
    }

    let mut tensor = TensorMap::new(Arc::new(filtered_keys), blocks)?;

    // Load info.json if present
    let info_path = String::from("info.json");
    if archive.file_names().any(|name| name == info_path) {
        let mut info_file = archive.by_name(&info_path).map_err(|e| (info_path, e))?;
        let mut info = String::new();
        info_file.read_to_string(&mut info)?;
        let info = jzon::parse(&info).map_err(|e| Error::Serialization(e.to_string()))?;
        let info = info.as_object().ok_or_else(|| {
            Error::Serialization("'info.json' should contain an object".into())
        })?;

        for (key, value) in info.iter() {
            let value = value.as_str().ok_or_else(|| {
                Error::Serialization("values in 'info.json' should be strings".into())
            })?;
            tensor.add_info(
                key,
                ConstCString::new(
                    CString::new(value).expect("value in 'info.json' should not contain a NUL byte"),
                ),
            );
        }
    }

    Ok(tensor)
}

/// Build a new `Labels` containing only the rows at the given indices.
fn labels_subset(labels: &Labels, indices: &[usize]) -> Result<Labels, Error> {
    let names_owned = labels.names();
    let names: Vec<&str> = names_owned.iter().map(|s| &**s).collect();
    if names.is_empty() {
        return Labels::new(&names, Vec::new());
    }
    let mut values = Vec::with_capacity(indices.len() * names.len());
    for &idx in indices {
        values.extend_from_slice(&labels[idx]);
    }
    // Indices come from select() or sequential iteration → uniqueness guaranteed
    unsafe { Labels::new_unchecked_uniqueness(&names, values) }
}

/// Parse the NPY header from a ZIP entry within a memory-mapped file,
/// returning the shape, element size in bytes, and the byte offset of the
/// raw data within the mmap.
///
/// Requires STORED compression and native endianness.
fn parse_npy_entry_metadata(
    archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
    mmap: &[u8],
    path: &str,
) -> Result<(Vec<usize>, usize, usize), Error> {
    let entry = archive.by_name(path).map_err(|e| (path.to_string(), e))?;

    if entry.compression() != zip::CompressionMethod::Stored {
        return Err(Error::Serialization(format!(
            "entry '{}' uses compression {:?}, but partial loading requires STORED entries",
            path, entry.compression()
        )));
    }

    let entry_size = entry.size() as usize;
    let data_start = entry.data_start() as usize;
    drop(entry);

    if data_start + entry_size > mmap.len() {
        return Err(Error::Serialization(format!(
            "entry '{}' extends beyond the end of the file", path
        )));
    }

    let npy_bytes = &mmap[data_start..data_start + entry_size];
    let (header, npy_header_len) = Header::from_slice(npy_bytes)?;

    if header.fortran_order {
        return Err(Error::Serialization(
            "data cannot be loaded from fortran-order arrays".into(),
        ));
    }

    let descr = if let DataType::Scalar(s) = &header.type_descriptor {
        s.as_str()
    } else {
        return Err(Error::Serialization(
            "structured arrays are not supported for partial loading".into(),
        ));
    };

    let (_code, bits, endian) = npy_descr_to_dtype(descr)?;

    match endian {
        Endianness::Native => {}
        Endianness::Little => {
            if cfg!(target_endian = "big") {
                return Err(Error::Serialization(
                    "partial loading requires native endianness".into(),
                ));
            }
        }
        Endianness::Big => {
            if cfg!(target_endian = "little") {
                return Err(Error::Serialization(
                    "partial loading requires native endianness".into(),
                ));
            }
        }
    }

    let elem_size = (bits as usize) / 8;
    let raw_data_offset = data_start + npy_header_len;

    Ok((header.shape, elem_size, raw_data_offset))
}

/// Copy selected rows/columns from mmap bytes into a new array via DLPack.
///
/// `src_shape` is the full shape `[n_samples, comp..., n_properties]`.
/// `sample_indices` maps new_row → old_row.
/// `prop_indices` maps new_col → old_col (or `None` to select all properties).
fn gather_selected_data<F>(
    mmap: &[u8],
    raw_data_offset: usize,
    elem_size: usize,
    src_shape: &[usize],
    sample_indices: &[usize],
    prop_indices: Option<&[usize]>,
    create_array: &F,
) -> Result<mts_array_t, Error>
where
    F: Fn(Vec<usize>) -> Result<mts_array_t, Error>
{
    let src_n_props = *src_shape.last().unwrap();
    let n_components: usize = if src_shape.len() > 2 {
        src_shape[1..src_shape.len() - 1].iter().product()
    } else {
        1
    };
    let src_row_bytes = n_components * src_n_props * elem_size;

    let new_n_samples = sample_indices.len();
    let new_n_props = prop_indices.map_or(src_n_props, |p| p.len());

    let mut output_shape = Vec::with_capacity(src_shape.len());
    output_shape.push(new_n_samples);
    if src_shape.len() > 2 {
        output_shape.extend_from_slice(&src_shape[1..src_shape.len() - 1]);
    }
    output_shape.push(new_n_props);

    let output = create_array(output_shape)?;

    if new_n_samples == 0 || new_n_props == 0 {
        return Ok(output);
    }

    let device = DLDevice::cpu();
    let version = DLPackVersion::current();
    let mut dl_tensor = output.as_dlpack(device, None, version)?;
    let tensor_ref = dl_tensor.as_mut();

    let mut view: ndarray::ArrayViewMutD<f64> = tensor_ref.try_into()
        .map_err(|e| Error::Serialization(format!(
            "load_partial requires f64 arrays from create_array callback: {}", e
        )))?;

    let view_slice = view.as_slice_mut().ok_or_else(|| {
        Error::Serialization("output array from create_array is not C-contiguous".into())
    })?;

    let dst_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            view_slice.as_mut_ptr().cast::<u8>(),
            view_slice.len() * std::mem::size_of::<f64>(),
        )
    };

    if let Some(prop_idx) = prop_indices {
        let dst_row_bytes = n_components * new_n_props * elem_size;

        for (new_row, &old_row) in sample_indices.iter().enumerate() {
            let src_offset = raw_data_offset + old_row * src_row_bytes;
            let row_buf = &mmap[src_offset..src_offset + src_row_bytes];

            let dst_row_off = new_row * dst_row_bytes;
            for comp in 0..n_components {
                let src_comp_off = comp * src_n_props * elem_size;
                let dst_comp_off = dst_row_off + comp * new_n_props * elem_size;
                for (new_col, &old_col) in prop_idx.iter().enumerate() {
                    let src_off = src_comp_off + old_col * elem_size;
                    let dst_off = dst_comp_off + new_col * elem_size;
                    dst_bytes[dst_off..dst_off + elem_size]
                        .copy_from_slice(&row_buf[src_off..src_off + elem_size]);
                }
            }
        }
    } else {
        // All properties: memcpy full rows from mmap
        for (new_row, &old_row) in sample_indices.iter().enumerate() {
            let src_offset = raw_data_offset + old_row * src_row_bytes;
            let dst_offset = new_row * src_row_bytes;
            dst_bytes[dst_offset..dst_offset + src_row_bytes]
                .copy_from_slice(&mmap[src_offset..src_offset + src_row_bytes]);
        }
    }

    Ok(output)
}

/// Read a single block with partial sample/property selection.
fn read_partial_block<F>(
    archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
    mmap: &[u8],
    prefix: &str,
    samples_sel: Option<&Labels>,
    properties_sel: Option<&Labels>,
    parent_properties: Option<(Arc<Labels>, Option<&[usize]>)>,
    create_array: &F,
) -> Result<TensorBlock, Error>
where
    F: Fn(Vec<usize>) -> Result<mts_array_t, Error>
{
    let samples_path = format!("{}samples.npy", prefix);
    let samples_file = archive.by_name(&samples_path).map_err(|e| (samples_path, e))?;
    let all_samples = load_labels(samples_file)?;

    let values_path = format!("{}values.npy", prefix);
    let (src_shape, elem_size, raw_data_offset) =
        parse_npy_entry_metadata(archive, mmap, &values_path)?;

    let sample_indices = if let Some(sel) = samples_sel {
        let mut selected = vec![0i64; all_samples.count()];
        let n = all_samples.select(sel, &mut selected)?;
        selected[..n].iter().map(|&i| i as usize).collect::<Vec<_>>()
    } else {
        (0..all_samples.count()).collect()
    };

    let (new_properties, prop_indices): (Arc<Labels>, Option<Vec<usize>>) =
        if let Some((ref parent_props, ref parent_prop_idx)) = parent_properties {
            (parent_props.clone(), parent_prop_idx.map(|p| p.to_vec()))
        } else {
            let props_path = format!("{}properties.npy", prefix);
            let props_file = archive.by_name(&props_path).map_err(|e| (props_path, e))?;
            let all_props = load_labels(props_file)?;

            if let Some(sel) = properties_sel {
                let mut selected = vec![0i64; all_props.count()];
                let n = all_props.select(sel, &mut selected)?;
                let idx: Vec<usize> = selected[..n].iter().map(|&i| i as usize).collect();
                let filtered = labels_subset(&all_props, &idx)?;
                (Arc::new(filtered), Some(idx))
            } else {
                (Arc::new(all_props), None)
            }
        };

    let mut components = Vec::new();
    for i in 0..(src_shape.len() - 2) {
        let comp_path = format!("{}components/{}.npy", prefix, i);
        let comp_file = archive.by_name(&comp_path).map_err(|e| (comp_path, e))?;
        components.push(Arc::new(load_labels(comp_file)?));
    }

    let data = gather_selected_data(
        mmap,
        raw_data_offset,
        elem_size,
        &src_shape,
        &sample_indices,
        prop_indices.as_deref(),
        create_array,
    )?;

    let new_samples = Arc::new(labels_subset(&all_samples, &sample_indices)?);
    let mut block = TensorBlock::new(data, new_samples, components, new_properties.clone())?;

    if parent_properties.is_none() {
        let mut parameters = HashSet::new();
        let gradient_prefix = format!("{}gradients/", prefix);
        for name in archive.file_names() {
            if name.starts_with(&gradient_prefix) && name.ends_with("/samples.npy") {
                let (_, parameter) = name.split_at(gradient_prefix.len());
                let parameter = parameter.split('/').next()
                    .expect("could not find gradient parameter");
                parameters.insert(parameter.to_string());
            }
        }

        for parameter in &parameters {
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
    }

    Ok(block)
}

/// Read a gradient block with correct sample reindexing and property filtering.
///
/// Gradient samples have `["sample", ...]` where column 0 indexes parent
/// samples. We build a map from old parent sample index → new sequential
/// index, filter gradient samples where `entry[0]` is in the map, and
/// replace `entry[0]` with the new index.
fn read_partial_gradient<F>(
    archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
    mmap: &[u8],
    prefix: &str,
    parent_sample_indices: &[usize],
    new_properties: &Arc<Labels>,
    prop_indices: Option<&[usize]>,
    create_array: &F,
) -> Result<TensorBlock, Error>
where
    F: Fn(Vec<usize>) -> Result<mts_array_t, Error>
{
    let mut parent_map: HashMap<i32, i32> = HashMap::new();
    for (new_idx, &old_idx) in parent_sample_indices.iter().enumerate() {
        parent_map.insert(old_idx as i32, new_idx as i32);
    }

    let samples_path = format!("{}samples.npy", prefix);
    let samples_file = archive.by_name(&samples_path).map_err(|e| (samples_path, e))?;
    let grad_samples = load_labels(samples_file)?;

    let values_path = format!("{}values.npy", prefix);
    let (src_shape, elem_size, raw_data_offset) =
        parse_npy_entry_metadata(archive, mmap, &values_path)?;

    let mut components = Vec::new();
    for i in 0..(src_shape.len() - 2) {
        let comp_path = format!("{}components/{}.npy", prefix, i);
        let comp_file = archive.by_name(&comp_path).map_err(|e| (comp_path, e))?;
        components.push(Arc::new(load_labels(comp_file)?));
    }

    let grad_names = grad_samples.names();
    let grad_names_ref: Vec<&str> = grad_names.iter().map(|s| &**s).collect();
    let mut kept_grad_indices = Vec::new();
    let mut new_grad_values = Vec::new();

    for (i, entry) in grad_samples.iter().enumerate() {
        let parent_sample_idx = entry[0].i32();
        if let Some(&new_parent_idx) = parent_map.get(&parent_sample_idx) {
            kept_grad_indices.push(i);
            new_grad_values.push(LabelValue::from(new_parent_idx));
            for &val in &entry[1..] {
                new_grad_values.push(val);
            }
        }
    }

    let new_grad_samples = Arc::new(
        unsafe { Labels::new_unchecked_uniqueness(&grad_names_ref, new_grad_values)? }
    );

    let data = gather_selected_data(
        mmap,
        raw_data_offset,
        elem_size,
        &src_shape,
        &kept_grad_indices,
        prop_indices,
        create_array,
    )?;

    TensorBlock::new(data, new_grad_samples, components, new_properties.clone())
}
