use std::io::{BufReader, Read};
use std::sync::Arc;
use std::ffi::CString;

use zip::{ZipArchive, ZipWriter};

use crate::utils::ConstCString;
use crate::{TensorMap, Error, mts_array_t};

use super::PathOrBuffer;
use super::labels::{load_labels, save_labels};
use super::block::{read_single_block, write_single_block};


/// Check if the file/buffer in `data` looks like it could contain a serialized
/// `TensorMap`.
///
/// We check if the file seems to be a ZIP file containing `keys.npy`
pub fn looks_like_tensormap_data(mut data: PathOrBuffer) -> bool {
    match data {
        PathOrBuffer::Path(path) => {
            match std::fs::File::open(path) {
                Ok(file) => {
                    let mut buffer = BufReader::new(file);
                    return looks_like_tensormap_data(PathOrBuffer::Buffer(&mut buffer));
                },
                Err(_) => { return false; }
            }
        },
        PathOrBuffer::Buffer(ref mut buffer) => {
            match ZipArchive::new(buffer) {
                Ok(mut archive) => {
                    return archive.by_name("keys.npy").is_ok()
                }
                Err(_) => { return false; }
            }
        },
    }
}


/// Load the serialized tensor map from the given path.
///
/// Arrays for the values and gradient data will be created with the given
/// `create_array` callback, and filled by this function with the corresponding
/// data.
///
/// See the C API documentation for more information on the file format.
pub fn load<R, F>(reader: R, create_array: F) -> Result<TensorMap, Error>
    where R: std::io::Read + std::io::Seek,
          F: Fn(Vec<usize>) -> Result<mts_array_t, Error>
{
    let mut archive = ZipArchive::new(reader).map_err(|e| ("<root>".into(), e))?;

    let path = String::from("keys.npy");
    let keys = load_labels(archive.by_name(&path).map_err(|e| (path, e))?)?;

    let mut blocks = Vec::new();
    for block_i in 0..keys.count() {
        blocks.push(read_single_block(
            &mut archive,
            &format!("blocks/{}/", block_i),
            None,
            &create_array,
        )?,);
    }

    let mut tensor = TensorMap::new(Arc::new(keys), blocks)?;

    // Load info.json, if it exists
    let path = String::from("info.json");
    if archive.index_for_name(&path).is_some() {
        let mut info_file = archive.by_name(&path).map_err(|e| (path, e))?;

        let mut info = String::new();
        info_file.read_to_string(&mut info)?;
        let info = jzon::parse(&info).map_err(|e| Error::Serialization(e.to_string()))?;
        let info = info.as_object().ok_or_else(|| Error::Serialization("'info.json' should contain an object".into()))?;

        for (key, value) in info.iter() {
            let value = value.as_str().ok_or_else(|| Error::Serialization("values in 'info.json' should be strings".into()))?;
            tensor.add_info(key, ConstCString::new(CString::new(value).expect("value in 'info.json' should not contain a NUL byte")));
        }
    }

    return Ok(tensor);
}


/// Save the given tensor to a file (or any other writer).
///
/// The format consists of a zip archive containing NPY files and one JSON file
/// for the global metadata. The recomended file extension when saving data is
/// `.mts`, to prevent confusion with generic `.npz` files.
pub fn save<W: std::io::Write + std::io::Seek>(writer: W, tensor: &TensorMap) -> Result<(), Error> {
    let mut archive = ZipWriter::new(writer);

    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .large_file(true)
        .last_modified_time(zip::DateTime::from_date_and_time(2000, 1, 1, 0, 0, 0).expect("invalid datetime"));

    let path = String::from("keys.npy");
    archive.start_file(&path, options).map_err(|e| (path, e))?;
    save_labels(&mut archive, tensor.keys())?;

    for (block_i, block) in tensor.blocks().iter().enumerate() {
        write_single_block(&mut archive, &format!("blocks/{}/", block_i), true, block)?;
    }

    let mut info = jzon::JsonValue::new_object();
    for (key, value) in tensor.info() {
        info[key] = jzon::JsonValue::from(value.as_str());
    }
    if !info.is_empty() {
        let path = String::from("info.json");
        archive.start_file(&path, options).map_err(|e| (path, e))?;
        info.write_pretty(&mut archive, 2)?;
    }

    archive.finish().map_err(|e| ("<root>".into(), e))?;

    return Ok(());
}
