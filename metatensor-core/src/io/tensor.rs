use std::io::BufReader;
use std::sync::Arc;

use zip::{ZipArchive, ZipWriter};

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
/// `TensorMap` are serialized using numpy's `.npz` format, i.e. a ZIP file
/// without compression (storage method is STORED), where each file is stored as
/// a `.npy` array. Both the ZIP and NPY format are well documented:
///
/// - ZIP: <https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT>
/// - NPY: <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>
///
/// We add other restriction on top of these formats when saving/loading data.
/// First, `Labels` instances are saved as structured array, see the `labels`
/// module for more information. Only 32-bit integers are supported for Labels,
/// and only 64-bit floats are supported for data (values and gradients).
///
/// Second, the path of the files in the archive also carry meaning. The keys of
/// the `TensorMap` are stored in `/keys.npy`, and then different blocks are
/// stored as
///
/// ```bash
/// /  blocks / <block_id>  / samples.npy
///                         / components  / 0.npy
///                                       / <...>.npy
///                                       / <n_components>.npy
///                         / properties.npy
///                         / values.npy
///
///                         # optional sections for gradients, one by parameter
///                         /   gradients / <parameter> / samples.npy
///                                                     /   components  / 0.npy
///                                                                     / <...>.npy
///                                                                     / <n_components>.npy
///                                                     /   values.npy
/// ```
pub fn load<R, F>(reader: R, create_array: F) -> Result<TensorMap, Error>
    where R: std::io::Read + std::io::Seek,
          F: Fn(Vec<usize>) -> Result<mts_array_t, Error>
{
    let mut archive = ZipArchive::new(reader).map_err(|e| ("<root>".into(), e))?;

    let path = String::from("keys.npy");
    let keys = load_labels(archive.by_name(&path).map_err(|e| (path, e))?)?;

    if archive.by_name("blocks/0/values/data.npy").is_ok() {
        return Err(Error::Serialization(
            "trying to load a file in the old metatensor format, please convert \
            it to the new format first using the script at \
            https://github.com/metatensor/metatensor/blob/master/python/scripts/convert-metatensor-npz.py
            ".into()
        ));
    }

    let mut blocks = Vec::new();
    for block_i in 0..keys.count() {
        blocks.push(read_single_block(
            &mut archive,
            &format!("blocks/{}/", block_i),
            None,
            &create_array,
        )?,);
    }

    return TensorMap::new(Arc::new(keys), blocks);
}


/// Save the given tensor to a file (or any other writer).
///
/// The format used is documented in the [`load`] function, and is based on
/// numpy's NPZ format (i.e. zip archive containing NPY files).
pub fn save<W: std::io::Write + std::io::Seek>(writer: W, tensor: &TensorMap) -> Result<(), Error> {
    let mut archive = ZipWriter::new(writer);

    let options = zip::write::FileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .large_file(true)
        .last_modified_time(zip::DateTime::from_date_and_time(2000, 1, 1, 0, 0, 0).expect("invalid datetime"));

    let path = String::from("keys.npy");
    archive.start_file(&path, options).map_err(|e| (path, e))?;
    save_labels(&mut archive, tensor.keys())?;

    for (block_i, block) in tensor.blocks().iter().enumerate() {
        write_single_block(&mut archive, &format!("blocks/{}/", block_i), true, block)?;
    }

    archive.finish().map_err(|e| ("<root>".into(), e))?;

    return Ok(());
}
