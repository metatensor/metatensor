use std::collections::HashSet;
use std::ffi::CString;
use std::io::Read;
use std::sync::Arc;

use memmap2::Mmap;
use zip::ZipArchive;

use dlpk::sys::DLDataType;

use super::labels::load_labels;
use super::Endianness;
use super::block::npy_descr_to_dtype;
use super::npy_header::{Header, DataType};

use crate::utils::ConstCString;
use crate::{TensorMap, TensorBlock, Labels, Error, mts_array_t};

/// Load a `TensorMap` from the file at the given path using memory mapping.
///
/// This provides zero-copy loading for data arrays: instead of reading and
/// copying the array data, the file is memory-mapped and the array data
/// points directly into the mapped region. This can significantly reduce
/// memory usage and loading time for large datasets.
///
/// Labels (samples, components, properties) are still loaded normally since
/// they need to be owned by the `Labels` struct.
///
/// The file must use the STORED (uncompressed) ZIP format, which is the
/// default when saving with metatensor.
pub fn load_mmap<F>(path: &str, create_array: F) -> Result<TensorMap, Error>
where
    F: Fn(Vec<usize>, DLDataType, *const std::os::raw::c_void, usize, Arc<Mmap>) -> Result<mts_array_t, Error>
{
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| Error::Io(e))?;
    let mmap = Arc::new(mmap);

    let cursor = std::io::Cursor::new(mmap.as_ref().as_ref());
    let mut archive = ZipArchive::new(cursor).map_err(|e| ("<root>".into(), e))?;

    let path_str = String::from("keys.npy");
    let keys = load_labels(archive.by_name(&path_str).map_err(|e| (path_str, e))?)?;

    let mut blocks = Vec::new();
    for block_i in 0..keys.count() {
        let prefix = format!("blocks/{}/", block_i);
        let block = read_mmap_block(&mut archive, &mmap, &prefix, None, &create_array)?;
        blocks.push(block);
    }

    let mut tensor = TensorMap::new(Arc::new(keys), blocks)?;

    // Load info.json, if it exists
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

/// Load a `TensorBlock` from the file at the given path using memory mapping.
pub fn load_block_mmap<F>(path: &str, create_array: F) -> Result<TensorBlock, Error>
where
    F: Fn(Vec<usize>, DLDataType, *const std::os::raw::c_void, usize, Arc<Mmap>) -> Result<mts_array_t, Error>
{
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| Error::Io(e))?;
    let mmap = Arc::new(mmap);

    let cursor = std::io::Cursor::new(mmap.as_ref().as_ref());
    let mut archive = ZipArchive::new(cursor).map_err(|e| ("<root>".into(), e))?;

    read_mmap_block(&mut archive, &mmap, "", None, &create_array)
}

fn read_mmap_block<F>(
    archive: &mut ZipArchive<std::io::Cursor<&[u8]>>,
    mmap: &Arc<Mmap>,
    prefix: &str,
    properties: Option<Arc<Labels>>,
    create_array: &F,
) -> Result<TensorBlock, Error>
where
    F: Fn(Vec<usize>, DLDataType, *const std::os::raw::c_void, usize, Arc<Mmap>) -> Result<mts_array_t, Error>
{
    // Load values array via mmap
    let values_path = format!("{}values.npy", prefix);
    let (data, shape) = read_mmap_data(archive, mmap, &values_path, create_array)?;

    // Load labels normally
    let samples_path = format!("{}samples.npy", prefix);
    let samples_file = archive.by_name(&samples_path).map_err(|e| (samples_path, e))?;
    let samples = Arc::new(load_labels(samples_file)?);

    let mut components = Vec::new();
    for i in 0..(shape.len() - 2) {
        let comp_path = format!("{}components/{}.npy", prefix, i);
        let comp_file = archive.by_name(&comp_path).map_err(|e| (comp_path, e))?;
        components.push(Arc::new(load_labels(comp_file)?));
    }

    let properties = if let Some(ref properties) = properties {
        properties.clone()
    } else {
        let props_path = format!("{}properties.npy", prefix);
        let props_file = archive.by_name(&props_path).map_err(|e| (props_path, e))?;
        Arc::new(load_labels(props_file)?)
    };

    let mut block = TensorBlock::new(data, samples, components, properties.clone())?;

    // Find and load gradients
    let mut parameters = HashSet::new();
    let gradient_prefix = format!("{}gradients/", prefix);
    for name in archive.file_names() {
        if name.starts_with(&gradient_prefix) && name.ends_with("/samples.npy") {
            let (_, parameter) = name.split_at(gradient_prefix.len());
            let parameter = parameter.split('/').next().expect("could not find gradient parameter");
            parameters.insert(parameter.to_string());
        }
    }

    for parameter in &parameters {
        let gradient = read_mmap_block(
            archive,
            mmap,
            &format!("{}gradients/{}/", prefix, parameter),
            Some(properties.clone()),
            create_array,
        )?;
        block.add_gradient(parameter, gradient)?;
    }

    Ok(block)
}

fn read_mmap_data<F>(
    archive: &mut ZipArchive<std::io::Cursor<&[u8]>>,
    mmap: &Arc<Mmap>,
    path: &str,
    create_array: &F,
) -> Result<(mts_array_t, Vec<usize>), Error>
where
    F: Fn(Vec<usize>, DLDataType, *const std::os::raw::c_void, usize, Arc<Mmap>) -> Result<mts_array_t, Error>
{
    // Get the entry to find its data offset and verify it's STORED
    let entry = archive.by_name(path).map_err(|e| (path.to_string(), e))?;

    if entry.compression() != zip::CompressionMethod::Stored {
        return Err(Error::Serialization(format!(
            "entry '{}' uses compression method {:?}, but mmap loading requires STORED (uncompressed) entries",
            path, entry.compression()
        )));
    }

    let entry_size = entry.size() as usize;
    let data_start = entry.data_start() as usize;

    // Drop the entry to release the borrow on archive
    drop(entry);

    if data_start + entry_size > mmap.len() {
        return Err(Error::Serialization(format!(
            "entry '{}' extends beyond the end of the file", path
        )));
    }

    // Parse the NPY header from the mmap slice
    let npy_bytes = &mmap[data_start..data_start + entry_size];
    let (header, npy_header_len) = Header::from_slice(npy_bytes)?;

    if header.fortran_order {
        return Err(Error::Serialization(
            "data can not be loaded from fortran-order arrays".into(),
        ));
    }

    let descr = if let DataType::Scalar(s) = &header.type_descriptor {
        s.as_str()
    } else {
        return Err(Error::Serialization(
            "structured arrays are not supported for mmap loading".into(),
        ));
    };

    let (code, bits, endian) = npy_descr_to_dtype(descr)?;

    // Verify endianness matches native
    match endian {
        Endianness::Native => {}
        Endianness::Little => {
            if cfg!(target_endian = "big") {
                return Err(Error::Serialization(
                    "mmap loading requires native endianness; file has little-endian data on a big-endian system".into(),
                ));
            }
        }
        Endianness::Big => {
            if cfg!(target_endian = "little") {
                return Err(Error::Serialization(
                    "mmap loading requires native endianness; file has big-endian data on a little-endian system".into(),
                ));
            }
        }
    }

    let shape = header.shape;
    let num_elements: usize = shape.iter().product();
    let element_bytes = (bits as usize / 8) * 1; // lanes = 1
    let raw_data_offset = data_start + npy_header_len;
    let data_len = num_elements * element_bytes;

    if raw_data_offset + data_len > mmap.len() {
        return Err(Error::Serialization(format!(
            "NPY data in '{}' extends beyond the end of the file", path
        )));
    }

    let dl_dtype = DLDataType {
        code,
        bits,
        lanes: 1,
    };

    let array = create_array(
        shape.clone(),
        dl_dtype,
        unsafe { mmap.as_ptr().add(raw_data_offset) }.cast(),
        data_len,
        Arc::clone(mmap),
    )?;

    Ok((array, shape))
}
