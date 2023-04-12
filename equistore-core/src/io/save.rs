use byteorder::{WriteBytesExt, NativeEndian};
use zip::{ZipWriter, DateTime};

use crate::{TensorMap, TensorBlock, Error, eqs_array_t};

use super::npy_header::{Header, DataType};
use super::labels::write_npy_labels;


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
        write_block(&mut archive, &format!("blocks/{}", block_i), true, block)?;
    }

    archive.finish().map_err(|e| ("<root>".into(), e))?;

    return Ok(());
}

fn write_block<W: std::io::Write + std::io::Seek>(
    archive: &mut ZipWriter<W>,
    prefix: &str,
    values: bool,
    block: &TensorBlock,
) -> Result<(), Error> {
    let options = zip::write::FileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .large_file(true)
        .last_modified_time(DateTime::from_date_and_time(2000, 1, 1, 0, 0, 0).expect("invalid datetime"));

    let path = format!("{}/values.npy", prefix);
    archive.start_file(&path, options).map_err(|e| (path, e))?;
    write_data(archive, &block.values)?;

    let path = format!("{}/samples.npy", prefix);
    archive.start_file(&path, options).map_err(|e| (path, e))?;
    write_npy_labels(archive, &block.samples)?;

    for (i, component) in block.components.iter().enumerate() {
        let path = format!("{}/components/{}.npy", prefix, i);
        archive.start_file(&path, options).map_err(|e| (path, e))?;
        write_npy_labels(archive, component)?;
    }

    if values {
        let path = format!("{}/properties.npy", prefix);
        archive.start_file(&path, options).map_err(|e| (path.clone(), e))?;
        write_npy_labels(archive, &block.properties)?;
    }

    for (parameter, gradient) in block.gradients() {
        let prefix = format!("{}/gradients/{}", prefix, parameter);
        write_block(archive, &prefix, false, gradient)?;
    }

    Ok(())
}

// Write an array to the given writer, using numpy's NPY format
fn write_data<W: std::io::Write>(writer: &mut W, array: &eqs_array_t) -> Result<(), Error> {
    let type_descriptor = if cfg!(target_endian = "little") {
        "<f8"
    } else {
        ">f8"
    };

    let header = Header {
        type_descriptor: DataType::Scalar(type_descriptor.into()),
        fortran_order: false,
        shape: array.shape()?.to_vec(),
    };

    header.write(&mut *writer)?;

    for &value in array.data()? {
        writer.write_f64::<NativeEndian>(value)?;
    }

    return Ok(());
}
