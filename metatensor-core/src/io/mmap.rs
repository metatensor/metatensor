//! Memory-mapped loading of `TensorBlock` / `TensorMap` from
//! `.mts` files.
//!
//! Opens the file with `std::fs::File`, parses ZIP central-directory
//! and NPY-header bytes through a `BufReader<File>`, and dispatches
//! each array and label table to a caller-supplied `create_array` callback with
//! `(shape, dtype, file_offset)`. The caller decides how to
//! materialise the array: mmap-backed view, plain pread, GPU Direct
//! Storage upload, etc. The metatensor core itself never mmaps the
//! file -- only the binding sees the mapping (if it chooses to use
//! one) and is responsible for its lifecycle.
//!
//! Requirements on the input file:
//! - STORED (uncompressed) ZIP entries (this is what `mts_*_save` writes).
//! - Native byte order for all numeric arrays (mmap views and pread
//!   copies must be directly reinterpretable).
//! - `mts_*_save` aligns ZIP entry data so NPY payloads can be exported as
//!   typed mmap views. Callbacks must copy payloads whose offsets are not
//!   aligned for their dtype.

use std::ffi::CString;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

use zip::ZipArchive;

use dlpk::sys::DLDataType;

use super::labels::load_labels_mmap_from_npy_entry;
use super::parse_stored_npy_entry;

use crate::utils::ConstCString;
use crate::{mts_array_t, Error, Labels, TensorBlock, TensorMap};


/// Load a `TensorMap` from the file at the given path, using a callback to
/// create each value array from its file offset.
///
/// The implementation opens the file with `BufReader<File>`, parses
/// NPY headers to discover array shapes, dtypes, and byte offsets,
/// and then dispatches each array (labels + values + gradient values) to
/// `create_array(shape, dtype, file_offset)`.
///
/// `create_array` decides how to materialise the `mts_array_t`: it
/// can wrap a binding-side mmap as a zero-copy view, pread the bytes
/// into an owned buffer, or stream them to a GPU via GPU Direct
/// Storage. The byte length is `shape.iter().product() * (dtype.bits
/// / 8) * dtype.lanes`.
///
/// # File format constraints
/// - The file must use `STORED` (uncompressed) ZIP entries.
/// - Numeric arrays and label tables must use native byte order.
pub fn load_mmap<F>(path: &str, create_array: F) -> Result<TensorMap, Error>
where
    F: Fn(Vec<usize>, DLDataType, usize) -> Result<mts_array_t, Error>,
{
    let file = BufReader::new(File::open(path)?);
    let mut archive = ZipArchive::new(file).map_err(|e| ("<root>".into(), e))?;

    let keys_path = String::from("keys.npy");
    let keys = load_labels_mmap_from_npy_entry(&mut archive, &keys_path, &create_array)?;

    let mut blocks = Vec::with_capacity(keys.count());
    for block_i in 0..keys.count() {
        let prefix = format!("blocks/{}/", block_i);
        blocks.push(read_mmap_block(
            &mut archive,
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


/// Load a single `TensorBlock` from the file at the given path, using a
/// callback to create the value array from its file offset. See [`load_mmap`]
/// for callback semantics and file format constraints.
pub fn load_block_mmap<F>(path: &str, create_array: F) -> Result<TensorBlock, Error>
where
    F: Fn(Vec<usize>, DLDataType, usize) -> Result<mts_array_t, Error>,
{
    let file = BufReader::new(File::open(path)?);
    let mut archive = ZipArchive::new(file).map_err(|e| ("<root>".into(), e))?;

    read_mmap_block(&mut archive, "", None, &create_array)
}


// `properties` lets a gradient block reuse its parent block's
// `properties` Labels: TensorMap requires every gradient to share the
// parent's properties dimension, so the recursive `read_mmap_block`
// call for each gradient passes `Some(parent_properties)` and skips
// the `properties.npy` entry in the gradient's own subdirectory. A
// top-level block load passes `None` to read the entry from disk.
fn read_mmap_block<R, F>(
    archive: &mut ZipArchive<R>,
    prefix: &str,
    properties: Option<Arc<Labels>>,
    create_array: &F,
) -> Result<TensorBlock, Error>
where
    R: std::io::Read + std::io::Seek,
    F: Fn(Vec<usize>, DLDataType, usize) -> Result<mts_array_t, Error>,
{
    let values_path = format!("{}values.npy", prefix);
    let (shape, dl_dtype, file_offset) = parse_stored_npy_entry(archive, &values_path)?;
    let shape_len = shape.len();
    let data = create_array(shape, dl_dtype, file_offset)?;

    let samples_path = format!("{}samples.npy", prefix);
    let samples = Arc::new(load_labels_mmap_from_npy_entry(
        archive,
        &samples_path,
        create_array,
    )?);

    let mut components = Vec::new();
    for i in 0..shape_len.saturating_sub(2) {
        let path = format!("{}components/{}.npy", prefix, i);
        components.push(Arc::new(load_labels_mmap_from_npy_entry(
            archive,
            &path,
            create_array,
        )?));
    }

    let properties = if let Some(ref properties) = properties {
        properties.clone()
    } else {
        let path = format!("{}properties.npy", prefix);
        Arc::new(load_labels_mmap_from_npy_entry(
            archive,
            &path,
            create_array,
        )?)
    };

    let mut block = TensorBlock::new(data, samples, components, properties.clone())?;

    for parameter in &super::discover_gradient_parameters(archive, prefix) {
        let gradient = read_mmap_block(
            archive,
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
    use std::fs::File;
    use std::io::{Seek, SeekFrom};
    use std::sync::{Arc, Mutex};

    use byteorder::{NativeEndian, ReadBytesExt};

    use super::*;
    use crate::data::TestArray;
    use crate::labels::create_test_array_from_vec;

    const DATA_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data.mts");

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
            if dtype.code == dlpk::sys::DLDataTypeCode::kDLInt
                && dtype.bits == 32
                && dtype.lanes == 1
                && shape.len() == 2
            {
                let count = shape[0];
                let size = shape[1];
                let mut file = File::open(DATA_PATH)?;
                file.seek(SeekFrom::Start(file_offset as u64))?;
                let mut values = vec![0; count * size];
                file.read_i32_into::<NativeEndian>(&mut values)?;
                Ok(create_test_array_from_vec(values, count, size))
            } else {
                Ok(TestArray::new(shape))
            }
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
        // Two independent loads of the same file must produce identical
        // key/sample/property/gradient structure. Data equality is verified
        // at the binding layer (Python/Torch) where real arrays are
        // materialised; here we only have metadata-only TestArrays, which
        // is enough for structural cross-checking.
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
}
