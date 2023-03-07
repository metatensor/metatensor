use std::sync::Arc;

use byteorder::{LittleEndian, BigEndian, ReadBytesExt, WriteBytesExt, NativeEndian};
use py_literal::Value as PyValue;
use zip::{ZipArchive, ZipWriter, DateTime};

use crate::{TensorMap, Error, TensorBlock, eqs_array_t};


mod npy_header;
pub use self::npy_header::Header;

mod labels;
use self::labels::{read_npy_labels, write_npy_labels};

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
/// /  blocks / <block_id>  / values / samples.npy
///                         / values / components  / 0.npy
///                                                / <...>.npy
///                                                / <n_components>.npy
///                         / values / properties.npy
///                         / values / data.npy
///
///                         # optional sections for gradients, one by parameter
///                         /   gradients / <parameter> / samples.npy
///                                                     /   components  / 0.npy
///                                                                     / <...>.npy
///                                                                     / <n_components>.npy
///                                                     /   data.npy
/// ```
pub fn load<R, F>(reader: R, create_array: F) -> Result<TensorMap, Error>
    where R: std::io::Read + std::io::Seek,
          F: Fn(Vec<usize>) -> Result<eqs_array_t, Error>
{
    let mut archive = ZipArchive::new(reader).map_err(|e| ("<root>".into(), e))?;

    let path = String::from("keys.npy");
    let keys = read_npy_labels(archive.by_name(&path).map_err(|e| (path, e))?)?;

    let mut parameters = Vec::new();
    for name in archive.file_names() {
        if name.starts_with("blocks/0/gradients/") && name.ends_with("/data.npy") {
            let (_, parameter) = name.split_at(19);
            let (parameter, _) = parameter.split_at(parameter.len() - 9);
            parameters.push(parameter.to_string());
        }
    }

    let mut blocks = Vec::new();
    for block_i in 0..keys.count() {
        let path = format!("blocks/{}/values/data.npy", block_i);
        let data_file = archive.by_name(&path).map_err(|e| (path, e))?;
        let (data, shape) = read_data(data_file, &create_array)?;

        let path = format!("blocks/{}/values/samples.npy", block_i);
        let samples_file = archive.by_name(&path).map_err(|e| (path, e))?;
        let samples = Arc::new(read_npy_labels(samples_file)?);

        let mut components = Vec::new();
        for i in 0..(shape.len() - 2) {
            let path = format!("blocks/{}/values/components/{}.npy", block_i, i);
            let component_file = archive.by_name(&path).map_err(|e| (path, e))?;
            components.push(Arc::new(read_npy_labels(component_file)?));
        }

        let path = format!("blocks/{}/values/properties.npy", block_i);
        let properties_file = archive.by_name(&path).map_err(|e| (path, e))?;
        let properties = Arc::new(read_npy_labels(properties_file)?);

        let mut block = TensorBlock::new(data, samples, components, properties)?;

        for parameter in &parameters {
            let path = format!("blocks/{}/gradients/{}/data.npy", block_i, parameter);
            let data_file = archive.by_name(&path).map_err(|e| (path, e))?;
            let (data, shape) = read_data(data_file, &create_array)?;

            let path = format!("blocks/{}/gradients/{}/samples.npy", block_i, parameter);
            let samples_file = archive.by_name(&path).map_err(|e| (path, e))?;
            let samples = Arc::new(read_npy_labels(samples_file)?);

            let mut components = Vec::new();
            for i in 0..(shape.len() - 2) {
                let path = format!("blocks/{}/gradients/{}/components/{}.npy", block_i, parameter, i);
                let component_file = archive.by_name(&path).map_err(|e| (path, e))?;
                components.push(Arc::new(read_npy_labels(component_file)?));
            }

            block.add_gradient(parameter, data, samples, components)?;
        }

        blocks.push(block);
    }

    return TensorMap::new(keys, blocks);
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
        .last_modified_time(DateTime::from_date_and_time(2000, 1, 1, 0, 0, 0).expect("invalid datetime"));

    let path = String::from("keys.npy");
    archive.start_file(&path, options).map_err(|e| (path, e))?;
    write_npy_labels(&mut archive, tensor.keys())?;

    for (block_i, block) in tensor.blocks().iter().enumerate() {
        let path = format!("blocks/{}/values/data.npy", block_i);
        archive.start_file(&path, options).map_err(|e| (path, e))?;
        write_data(&mut archive, &block.values().data)?;

        let path = format!("blocks/{}/values/samples.npy", block_i);
        archive.start_file(&path, options).map_err(|e| (path, e))?;
        write_npy_labels(&mut archive, &block.values().samples)?;

        for (i, component) in block.values().components.iter().enumerate() {
            let path = format!("blocks/{}/values/components/{}.npy", block_i, i);
            archive.start_file(&path, options).map_err(|e| (path, e))?;
            write_npy_labels(&mut archive, component)?;
        }

        let path = format!("blocks/{}/values/properties.npy", block_i);
        archive.start_file(&path, options).map_err(|e| (path, e))?;
        write_npy_labels(&mut archive, &block.values().properties)?;

        for (parameter, gradient) in block.gradients() {
            let path = format!("blocks/{}/gradients/{}/data.npy", block_i, parameter);
            archive.start_file(&path, options).map_err(|e| (path, e))?;
            write_data(&mut archive, &gradient.data)?;

            let path = format!("blocks/{}/gradients/{}/samples.npy", block_i, parameter);
            archive.start_file(&path, options).map_err(|e| (path, e))?;
            write_npy_labels(&mut archive, &gradient.samples)?;

            for (i, component) in gradient.components.iter().enumerate() {
                let path = format!("blocks/{}/gradients/{}/components/{}.npy", block_i, parameter, i);
                archive.start_file(&path, options).map_err(|e| (path, e))?;
                write_npy_labels(&mut archive, component)?;
            }
        }
    }

    archive.finish().map_err(|e| ("<root>".into(), e))?;

    return Ok(());
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
        PyValue::String(s) if s == "<f8" => {
            reader.read_f64_into::<LittleEndian>(array.data_mut()?)?;
        }
        PyValue::String(s) if s == ">f8" => {
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

// returns an error if the given reader contains any more data
fn check_for_extra_bytes<R: std::io::Read>(reader: &mut R) -> Result<(), Error> {
    let extra = reader.read_to_end(&mut Vec::new())?;
    if extra == 0 {
        Ok(())
    } else {
        Err(Error::Serialization(format!("found {} extra bytes after the expected end of data", extra)))
    }
}

// Write an array to the given writer, using numpy's NPY format
fn write_data<W: std::io::Write>(writer: &mut W, array: &eqs_array_t) -> Result<(), Error> {
    let type_descriptor = if cfg!(target_endian = "little") {
        "'<f8'"
    } else {
        "'>f8'"
    };

    let header = Header {
        type_descriptor: type_descriptor.parse().expect("invalid dtype"),
        fortran_order: false,
        shape: array.shape()?.to_vec(),
    };

    header.write(&mut *writer)?;

    for &value in array.data()? {
        writer.write_f64::<NativeEndian>(value)?;
    }

    return Ok(());
}
