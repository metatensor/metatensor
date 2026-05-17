//! Memory-mapped loading of `TensorBlock` / `TensorMap` from `.mts` files.
//!
//! Parses NPY headers via mmap, then dispatches each array to a
//! caller-supplied `create_array` callback with `(shape, dtype, file_offset)`.
//! The caller decides how to materialise the array: mmap-backed view, plain
//! read, GPU Direct Storage upload, etc.
//!
//! Requirements on the input file:
//! - STORED (uncompressed) ZIP entries (this is what `mts_*_save` writes).
//! - Native byte order for all numeric arrays (mmap views must be directly
//!   reinterpretable).

use std::ffi::CString;
use std::io::Cursor;
use std::sync::Arc;

use memmap2::Mmap;
use zip::ZipArchive;

use dlpk::sys::DLDataType;

use super::labels::load_labels;
use super::parse_stored_npy_entry;

use crate::utils::ConstCString;
use crate::{mts_array_t, Error, Labels, TensorBlock, TensorMap};


/// Load a `TensorMap` from the file at the given path using memory mapping.
///
/// The implementation memory-maps the file, parses NPY headers to discover
/// array shapes, dtypes, and byte offsets, and then dispatches each array
/// (values + gradient values) to `create_array(shape, dtype, file_offset)`.
///
/// `create_array` decides how to materialise the `mts_array_t`: it can wrap
/// the mmap-ed bytes as a zero-copy view, copy them into an owned buffer,
/// or stream them to a GPU via GPU Direct Storage. The byte length is
/// `shape.iter().product() * (dtype.bits / 8) * dtype.lanes`.
///
/// Labels are loaded normally (decompressed into owned `Labels` values).
///
/// # File format constraints
/// - The file must use `STORED` (uncompressed) ZIP entries.
/// - Numeric arrays must use native byte order.
pub fn load_mmap<F>(path: &str, create_array: F) -> Result<TensorMap, Error>
where
    F: Fn(Vec<usize>, DLDataType, usize) -> Result<mts_array_t, Error>,
{
    let file = std::fs::File::open(path)?;
    // SAFETY: we treat the mmap as a read-only view; the underlying file is
    // owned by `file` for the duration of this call.
    let mmap = unsafe { Mmap::map(&file) }.map_err(Error::Io)?;

    let cursor = Cursor::new(mmap.as_ref());
    let mut archive = ZipArchive::new(cursor).map_err(|e| ("<root>".into(), e))?;

    let keys_path = String::from("keys.npy");
    let keys = load_labels(archive.by_name(&keys_path).map_err(|e| (keys_path, e))?)?;

    let mut blocks = Vec::with_capacity(keys.count());
    for block_i in 0..keys.count() {
        let prefix = format!("blocks/{}/", block_i);
        blocks.push(read_mmap_block(
            &mut archive,
            mmap.as_ref(),
            &prefix,
            None,
            &create_array,
        )?);
    }

    let mut tensor = TensorMap::new(Arc::new(keys), blocks)?;

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


/// Load a single `TensorBlock` from the file at the given path using memory
/// mapping. See [`load_mmap`] for callback semantics and file format
/// constraints.
pub fn load_block_mmap<F>(path: &str, create_array: F) -> Result<TensorBlock, Error>
where
    F: Fn(Vec<usize>, DLDataType, usize) -> Result<mts_array_t, Error>,
{
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(Error::Io)?;

    let cursor = Cursor::new(mmap.as_ref());
    let mut archive = ZipArchive::new(cursor).map_err(|e| ("<root>".into(), e))?;

    read_mmap_block(&mut archive, mmap.as_ref(), "", None, &create_array)
}


fn read_mmap_block<F>(
    archive: &mut ZipArchive<Cursor<&[u8]>>,
    mmap: &[u8],
    prefix: &str,
    properties: Option<Arc<Labels>>,
    create_array: &F,
) -> Result<TensorBlock, Error>
where
    F: Fn(Vec<usize>, DLDataType, usize) -> Result<mts_array_t, Error>,
{
    let values_path = format!("{}values.npy", prefix);
    let (shape, dl_dtype, file_offset) = parse_stored_npy_entry(archive, mmap, &values_path)?;
    let shape_len = shape.len();
    let data = create_array(shape, dl_dtype, file_offset)?;

    let samples_path = format!("{}samples.npy", prefix);
    let samples_file = archive.by_name(&samples_path).map_err(|e| (samples_path, e))?;
    let samples = Arc::new(load_labels(samples_file)?);

    let mut components = Vec::new();
    for i in 0..shape_len.saturating_sub(2) {
        let path = format!("{}components/{}.npy", prefix, i);
        let component_file = archive.by_name(&path).map_err(|e| (path, e))?;
        components.push(Arc::new(load_labels(component_file)?));
    }

    let properties = if let Some(ref properties) = properties {
        properties.clone()
    } else {
        let path = format!("{}properties.npy", prefix);
        let properties_file = archive.by_name(&path).map_err(|e| (path, e))?;
        Arc::new(load_labels(properties_file)?)
    };

    let mut block = TensorBlock::new(data, samples, components, properties.clone())?;

    for parameter in &super::discover_gradient_parameters(archive, prefix) {
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


#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::data::TestArray;

    const DATA_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data.mts");
    const BLOCK_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/block.mts");

    /// What metatensor handed back to the callback for one array.
    #[derive(Clone, Debug)]
    struct CallbackRecord {
        shape: Vec<usize>,
        dtype: DLDataType,
        file_offset: usize,
    }

    /// Build a callback that records the (shape, dtype, file_offset) it was
    /// invoked with and returns a metadata-only `TestArray`. Real bindings
    /// (numpy, torch, cuFile) would instead materialise an array at
    /// `file_offset`; the metadata-only flavour is enough to assert that
    /// metatensor is dispatching consistently.
    fn record_and_test_array(
        records: Arc<Mutex<Vec<CallbackRecord>>>,
    ) -> impl Fn(Vec<usize>, DLDataType, usize) -> Result<mts_array_t, Error> {
        move |shape, dtype, file_offset| {
            records.lock().unwrap().push(CallbackRecord {
                shape: shape.clone(),
                dtype,
                file_offset,
            });
            Ok(TestArray::new(shape))
        }
    }

    #[test]
    fn callback_metadata_is_consistent() {
        let records: Arc<Mutex<Vec<CallbackRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let cb = record_and_test_array(Arc::clone(&records));
        let _ = load_mmap(DATA_PATH, cb).expect("mmap load failed");

        let recs = records.lock().unwrap();
        assert!(!recs.is_empty(), "expected at least one callback invocation");
        for rec in recs.iter() {
            assert!(!rec.shape.is_empty(), "callback shape should be non-empty");
            assert!(rec.file_offset > 0, "file_offset should be inside the file");
            assert_eq!(rec.dtype.lanes, 1, "test fixture uses scalar dtype only");
        }
    }

    #[test]
    fn two_mmap_loads_are_structurally_identical() {
        // Two independent mmap loads of the same file must produce
        // identical key/sample/property/gradient structure. Data equality
        // is verified at the binding layer (Python/Torch) where real
        // arrays are materialised; here we only have metadata-only
        // TestArrays, which is enough for structural cross-checking.
        let records_a: Arc<Mutex<Vec<CallbackRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let a = load_mmap(DATA_PATH, record_and_test_array(Arc::clone(&records_a)))
            .expect("mmap load A failed");

        let records_b: Arc<Mutex<Vec<CallbackRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let b = load_mmap(DATA_PATH, record_and_test_array(Arc::clone(&records_b)))
            .expect("mmap load B failed");

        assert_eq!(a.keys().count(), b.keys().count());
        assert_eq!(a.keys().dimensions(), b.keys().dimensions());
        assert_eq!(records_a.lock().unwrap().len(), records_b.lock().unwrap().len());

        for (ba, bb) in a.blocks().iter().zip(b.blocks().iter()) {
            assert_eq!(ba.samples.count(), bb.samples.count());
            assert_eq!(ba.properties.count(), bb.properties.count());
            assert_eq!(ba.gradients().len(), bb.gradients().len());
        }
    }

    #[test]
    fn load_block_mmap_invokes_callback() {
        let records: Arc<Mutex<Vec<CallbackRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let _block = load_block_mmap(BLOCK_PATH, record_and_test_array(Arc::clone(&records)))
            .expect("mmap block load failed");

        let count = records.lock().unwrap().len();
        assert!(count >= 1, "expected at least one callback invocation, got {}", count);
    }
}
