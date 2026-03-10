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
use super::mmap_array::MmapArray;
use super::npy_header::{Header, DataType};

use crate::utils::ConstCString;
use crate::{TensorMap, TensorBlock, Labels, Error, mts_array_t};

/// Parsed metadata for a single NPY array entry within a ZIP archive.
pub(crate) struct ArrayFileInfo {
    pub shape: Vec<usize>,
    pub dl_dtype: DLDataType,
    /// Byte offset of the raw array data within the file.
    pub file_offset: usize,
    /// Byte length of the raw array data.
    pub data_len: usize,
}

/// Parse a NPY entry in the ZIP archive, returning its shape, dtype, and
/// file offset without creating any array. The mmap is used for fast
/// header parsing only.
fn parse_npy_entry(
    archive: &mut ZipArchive<std::io::Cursor<&[u8]>>,
    mmap: &Mmap,
    path: &str,
) -> Result<ArrayFileInfo, Error> {
    let entry = archive.by_name(path).map_err(|e| (path.to_string(), e))?;

    if entry.compression() != zip::CompressionMethod::Stored {
        return Err(Error::Serialization(format!(
            "entry '{}' uses compression method {:?}, but mmap loading requires STORED (uncompressed) entries",
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

    Ok(ArrayFileInfo {
        shape,
        dl_dtype,
        file_offset: raw_data_offset,
        data_len,
    })
}

/// Load labels (samples, components, properties) and find gradient
/// parameter names from the archive.
fn load_block_metadata(
    archive: &mut ZipArchive<std::io::Cursor<&[u8]>>,
    prefix: &str,
    ndim: usize,
    properties: Option<Arc<Labels>>,
) -> Result<(Arc<Labels>, Vec<Arc<Labels>>, Arc<Labels>, HashSet<String>), Error> {
    let samples_path = format!("{}samples.npy", prefix);
    let samples_file = archive.by_name(&samples_path).map_err(|e| (samples_path, e))?;
    let samples = Arc::new(load_labels(samples_file)?);

    let mut components = Vec::new();
    for i in 0..(ndim - 2) {
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

    let mut parameters = HashSet::new();
    let gradient_prefix = format!("{}gradients/", prefix);
    for name in archive.file_names() {
        if name.starts_with(&gradient_prefix) && name.ends_with("/samples.npy") {
            let (_, parameter) = name.split_at(gradient_prefix.len());
            let parameter = parameter.split('/').next().expect("could not find gradient parameter");
            parameters.insert(parameter.to_string());
        }
    }

    Ok((samples, components, properties, parameters))
}

// ============================================================================
// Default path: internal MmapArray (no callback)
// ============================================================================

/// Load a `TensorMap` from the file at the given path using memory mapping.
///
/// Data arrays are created internally as read-only mmap-backed arrays.
/// Labels are still loaded normally.
///
/// The file must use the STORED (uncompressed) ZIP format.
pub fn load_mmap(path: &str) -> Result<TensorMap, Error> {
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
        let block = read_mmap_block(&mut archive, &mmap, &prefix, None)?;
        blocks.push(block);
    }

    let mut tensor = TensorMap::new(Arc::new(keys), blocks)?;
    load_info(&mut archive, &mut tensor)?;

    Ok(tensor)
}

/// Load a `TensorBlock` from the file at the given path using memory mapping.
pub fn load_block_mmap(path: &str) -> Result<TensorBlock, Error> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| Error::Io(e))?;
    let mmap = Arc::new(mmap);

    let cursor = std::io::Cursor::new(mmap.as_ref().as_ref());
    let mut archive = ZipArchive::new(cursor).map_err(|e| ("<root>".into(), e))?;

    read_mmap_block(&mut archive, &mmap, "", None)
}

fn read_mmap_block(
    archive: &mut ZipArchive<std::io::Cursor<&[u8]>>,
    mmap: &Arc<Mmap>,
    prefix: &str,
    properties: Option<Arc<Labels>>,
) -> Result<TensorBlock, Error> {
    let values_path = format!("{}values.npy", prefix);
    let info = parse_npy_entry(archive, mmap, &values_path)?;
    let shape = info.shape.clone();

    let array = MmapArray::new(
        Arc::clone(mmap),
        info.file_offset,
        info.data_len,
        info.shape,
        info.dl_dtype,
    );
    let data = array.into_mts_array();

    let (samples, components, properties, gradient_params) =
        load_block_metadata(archive, prefix, shape.len(), properties)?;

    let mut block = TensorBlock::new(data, samples, components, properties.clone())?;

    for parameter in &gradient_params {
        let grad_prefix = format!("{}gradients/{}/", prefix, parameter);
        let gradient = read_mmap_block(archive, mmap, &grad_prefix, Some(properties.clone()))?;
        block.add_gradient(parameter, gradient)?;
    }

    Ok(block)
}

// ============================================================================
// Callback path: user-provided array creation from file offset
// ============================================================================

/// Load a `TensorMap` using a callback that receives file offsets.
///
/// For each data array, the callback receives the shape, DLPack dtype,
/// byte offset within the file, and byte length. The callback creates
/// the `mts_array_t` using whatever mechanism it prefers (mmap, GDS
/// direct-to-GPU read, pread, etc.).
///
/// The file is memory-mapped internally for fast ZIP/NPY header parsing
/// only; the callback is responsible for creating the actual data arrays.
pub fn load_mmap_with<F>(path: &str, create_array: F) -> Result<TensorMap, Error>
where
    F: Fn(&[usize], DLDataType, usize, usize) -> Result<mts_array_t, Error>
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
        let block = read_callback_block(&mut archive, &mmap, &prefix, None, &create_array)?;
        blocks.push(block);
    }

    let mut tensor = TensorMap::new(Arc::new(keys), blocks)?;
    load_info(&mut archive, &mut tensor)?;

    Ok(tensor)
}

/// Load a `TensorBlock` using a callback that receives file offsets.
pub fn load_block_mmap_with<F>(path: &str, create_array: F) -> Result<TensorBlock, Error>
where
    F: Fn(&[usize], DLDataType, usize, usize) -> Result<mts_array_t, Error>
{
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| Error::Io(e))?;
    let mmap = Arc::new(mmap);

    let cursor = std::io::Cursor::new(mmap.as_ref().as_ref());
    let mut archive = ZipArchive::new(cursor).map_err(|e| ("<root>".into(), e))?;

    read_callback_block(&mut archive, &mmap, "", None, &create_array)
}

fn read_callback_block<F>(
    archive: &mut ZipArchive<std::io::Cursor<&[u8]>>,
    mmap: &Arc<Mmap>,
    prefix: &str,
    properties: Option<Arc<Labels>>,
    create_array: &F,
) -> Result<TensorBlock, Error>
where
    F: Fn(&[usize], DLDataType, usize, usize) -> Result<mts_array_t, Error>
{
    let values_path = format!("{}values.npy", prefix);
    let info = parse_npy_entry(archive, mmap, &values_path)?;
    let shape = info.shape.clone();
    let data = create_array(&info.shape, info.dl_dtype, info.file_offset, info.data_len)?;

    let (samples, components, properties, gradient_params) =
        load_block_metadata(archive, prefix, shape.len(), properties)?;

    let mut block = TensorBlock::new(data, samples, components, properties.clone())?;

    for parameter in &gradient_params {
        let grad_prefix = format!("{}gradients/{}/", prefix, parameter);
        let gradient = read_callback_block(archive, mmap, &grad_prefix, Some(properties.clone()), create_array)?;
        block.add_gradient(parameter, gradient)?;
    }

    Ok(block)
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Load info.json if it exists in the archive.
fn load_info(
    archive: &mut ZipArchive<std::io::Cursor<&[u8]>>,
    tensor: &mut TensorMap,
) -> Result<(), Error> {
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};
    use std::sync::Mutex;

    const DATA_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data.mts");
    const BLOCK_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/block.mts");

    /// Record of a single callback invocation.
    #[derive(Debug, Clone)]
    struct CallbackRecord {
        shape: Vec<usize>,
        dtype: DLDataType,
        file_offset: usize,
        data_len: usize,
    }

    /// Helper: create a callback that records metadata and creates MmapArrays.
    fn make_recording_callback(
        path: &str,
        records: Arc<Mutex<Vec<CallbackRecord>>>,
    ) -> impl Fn(&[usize], DLDataType, usize, usize) -> Result<mts_array_t, Error> {
        let path = String::from(path);
        move |shape: &[usize], dtype: DLDataType, file_offset: usize, data_len: usize| {
            records.lock().unwrap().push(CallbackRecord {
                shape: shape.to_vec(),
                dtype,
                file_offset,
                data_len,
            });

            let mmap_file = std::fs::File::open(&path)?;
            let mmap = unsafe { Mmap::map(&mmap_file) }.map_err(Error::Io)?;
            let mmap = Arc::new(mmap);

            let array = MmapArray::new(mmap, file_offset, data_len, shape.to_vec(), dtype);
            Ok(array.into_mts_array())
        }
    }

    /// Test that the callback path receives correct metadata and produces
    /// a TensorMap identical to the default MmapArray path.
    #[test]
    fn callback_receives_correct_metadata() {
        let records: Arc<Mutex<Vec<CallbackRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let create_array = make_recording_callback(DATA_PATH, Arc::clone(&records));

        let callback_tensor = load_mmap_with(DATA_PATH, create_array).unwrap();
        let default_tensor = load_mmap(DATA_PATH).unwrap();

        // Verify callbacks were invoked
        let records = records.lock().unwrap();
        assert!(!records.is_empty(), "callback should have been called at least once");

        // Every callback should have non-empty shape and positive data_len
        for (i, rec) in records.iter().enumerate() {
            assert!(!rec.shape.is_empty(), "callback {i}: shape should not be empty");
            assert!(rec.data_len > 0, "callback {i}: data_len should be positive");
            assert!(rec.file_offset > 0, "callback {i}: file_offset should be positive");

            // Verify element count matches data_len
            let element_bytes = (rec.dtype.bits as usize / 8) * rec.dtype.lanes as usize;
            let num_elements: usize = rec.shape.iter().product();
            assert_eq!(
                num_elements * element_bytes,
                rec.data_len,
                "callback {i}: shape product * element_size should match data_len"
            );
        }

        // Compare callback result with default path
        assert_eq!(default_tensor.keys().count(), callback_tensor.keys().count());
        assert_eq!(default_tensor.keys().names(), callback_tensor.keys().names());

        for (db, cb) in default_tensor.blocks().iter().zip(callback_tensor.blocks().iter()) {
            assert_eq!(
                db.values.shape().unwrap(),
                cb.values.shape().unwrap(),
                "block values shape mismatch between default and callback"
            );

            assert_eq!(db.samples.count(), cb.samples.count());
            assert_eq!(db.properties.count(), cb.properties.count());
            assert_eq!(db.gradients().len(), cb.gradients().len());
        }
    }

    /// Test callback path for block loading.
    #[test]
    fn callback_block_loading() {
        let records: Arc<Mutex<Vec<CallbackRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let create_array = make_recording_callback(BLOCK_PATH, Arc::clone(&records));

        let callback_block = load_block_mmap_with(BLOCK_PATH, create_array).unwrap();
        let default_block = load_block_mmap(BLOCK_PATH).unwrap();

        // Callback should have been called for values + gradient values
        let count = records.lock().unwrap().len();
        assert!(count >= 2, "expected at least 2 callback calls (values + gradient), got {count}");

        assert_eq!(
            callback_block.values.shape().unwrap(),
            default_block.values.shape().unwrap()
        );
        assert_eq!(callback_block.samples.count(), default_block.samples.count());
        assert_eq!(callback_block.properties.count(), default_block.properties.count());
    }

    /// Test that callback receives f64 dtype for standard data arrays.
    #[test]
    fn callback_dtype_is_f64() {
        let records: Arc<Mutex<Vec<CallbackRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let create_array = make_recording_callback(BLOCK_PATH, Arc::clone(&records));

        let _ = load_block_mmap_with(BLOCK_PATH, create_array).unwrap();

        let records = records.lock().unwrap();
        assert!(!records.is_empty());
        for (i, rec) in records.iter().enumerate() {
            // kDLFloat = 2, 64 bits, 1 lane
            assert_eq!(rec.dtype.code as u32, 2, "callback {i}: expected kDLFloat (2)");
            assert_eq!(rec.dtype.bits, 64, "callback {i}: expected 64 bits");
            assert_eq!(rec.dtype.lanes, 1, "callback {i}: expected 1 lane");
        }
    }

    /// Test that file_offset values are valid by reading data from file
    /// at the reported offsets and comparing with mmap default path.
    #[test]
    fn callback_offsets_match_file_data() {
        let records: Arc<Mutex<Vec<CallbackRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let create_array = make_recording_callback(BLOCK_PATH, Arc::clone(&records));

        let _ = load_block_mmap_with(BLOCK_PATH, create_array).unwrap();

        // Verify each offset points to valid data in the file
        let file_len = std::fs::metadata(BLOCK_PATH).unwrap().len() as usize;
        let records = records.lock().unwrap();
        for (i, rec) in records.iter().enumerate() {
            assert!(
                rec.file_offset + rec.data_len <= file_len,
                "callback {i}: offset {} + len {} exceeds file size {}",
                rec.file_offset, rec.data_len, file_len
            );
        }

        // Read data from file at first callback's offset and verify it matches
        // the mmap view
        if let Some(first) = records.first() {
            let mut file = std::fs::File::open(BLOCK_PATH).unwrap();
            file.seek(SeekFrom::Start(first.file_offset as u64)).unwrap();
            let mut pread_data = vec![0u8; first.data_len];
            std::io::Read::read_exact(&mut file, &mut pread_data).unwrap();

            // The mmap default path should give us the same bytes
            let mmap_file = std::fs::File::open(BLOCK_PATH).unwrap();
            let mmap = unsafe { Mmap::map(&mmap_file) }.unwrap();
            let mmap_slice = &mmap[first.file_offset..first.file_offset + first.data_len];
            assert_eq!(pread_data, mmap_slice, "pread data should match mmap data");
        }
    }
}
