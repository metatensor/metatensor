use std::collections::HashSet;
use std::sync::Arc;

use byteorder::{LittleEndian, BigEndian, ReadBytesExt};
use zip::ZipArchive;

use crate::{TensorMap, TensorBlock, Labels, Error, eqs_array_t};

use super::check_for_extra_bytes;
use super::labels::read_npy_labels;
use super::npy_header::{Header, DataType};


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
          F: Fn(Vec<usize>) -> Result<eqs_array_t, Error>
{
    let mut archive = ZipArchive::new(reader).map_err(|e| ("<root>".into(), e))?;

    let path = String::from("keys.npy");
    let keys = read_npy_labels(archive.by_name(&path).map_err(|e| (path, e))?)?;

    // FIXME before merging: should we add a fallback?
    assert!(
        archive.by_name("blocks/0/values/data.npy").is_err(),
        "this is the old data format, TODO: fallback"
    );

    let mut blocks = Vec::new();
    for block_i in 0..keys.count() {
        blocks.push(read_block(
            &mut archive,
            &format!("blocks/{}", block_i),
            None,
            &create_array,
        )?,);
    }

    return TensorMap::new(keys, blocks);
}

#[allow(clippy::needless_pass_by_value)]
fn read_block<R, F>(
    archive: &mut ZipArchive<R>,
    prefix: &str,
    properties: Option<Arc<Labels>>,
    create_array: &F,
) -> Result<TensorBlock, Error>
    where R: std::io::Read + std::io::Seek,
          F: Fn(Vec<usize>) -> Result<eqs_array_t, Error>
{
    let path = format!("{}/values.npy", prefix);
    let data_file = archive.by_name(&path).map_err(|e| (path, e))?;
    let (data, shape) = read_data(data_file, &create_array)?;

    let path = format!("{}/samples.npy", prefix);
    let samples_file = archive.by_name(&path).map_err(|e| (path, e))?;
    let samples = Arc::new(read_npy_labels(samples_file)?);

    let mut components = Vec::new();
    for i in 0..(shape.len() - 2) {
        let path = format!("{}/components/{}.npy", prefix, i);
        let component_file = archive.by_name(&path).map_err(|e| (path, e))?;
        components.push(Arc::new(read_npy_labels(component_file)?));
    }

    let properties = if let Some(ref properties) = properties {
        properties.clone()
    } else {
        let path = format!("{}/properties.npy", prefix);
        let properties_file = archive.by_name(&path).map_err(|e| (path, e))?;
        Arc::new(read_npy_labels(properties_file)?)
    };

    let mut block = TensorBlock::new(data, samples, components, properties.clone())?;

    let mut parameters = HashSet::new();
    let gradient_prefix = format!("{}/gradients/", prefix);
    for name in archive.file_names() {
        if name.starts_with(&gradient_prefix) && name.ends_with("/samples.npy") {
            let (_, parameter) = name.split_at(gradient_prefix.len());
            let parameter = parameter.split('/').next().expect("could not find gradient parameter");
            parameters.insert(parameter.to_string());
        }
    }

    for parameter in &parameters {
        let gradient = read_block(
            archive,
            &format!("{}/gradients/{}", prefix, parameter),
            Some(properties.clone()),
            create_array
        )?;

        block.add_gradient(parameter, gradient)?;
    }

    return Ok(block);
}

// Read a data array from the given reader, using numpy's NPY format
fn read_data<R, F>(mut reader: R, create_array: &F) -> Result<(eqs_array_t, Vec<usize>), Error>
    where R: std::io::Read, F: Fn(Vec<usize>) -> Result<eqs_array_t, Error>
{
    let header = Header::from_reader(&mut reader)?;
    if header.fortran_order {
        return Err(Error::Serialization("data can not be loaded from fortran-order arrays".into()));
    }

    let shape = header.shape;
    let mut array = create_array(shape.clone())?;

    match header.type_descriptor {
        DataType::Scalar(s) if s == "<f8" => {
            reader.read_f64_into::<LittleEndian>(array.data_mut()?)?;
        }
        DataType::Scalar(s) if s == ">f8" => {
            reader.read_f64_into::<BigEndian>(array.data_mut()?)?;
        }
        _ => {
            return Err(Error::Serialization(format!(
                "unknown type for data array, expected 64-bit floating points, got {}",
                header.type_descriptor
            )));
        }
    }

    check_for_extra_bytes(&mut reader)?;

    return Ok((array, shape));
}
