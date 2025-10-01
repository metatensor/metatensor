use std::io::BufReader;
use std::collections::HashSet;
use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt, BigEndian, WriteBytesExt, NativeEndian};
use zip::{ZipArchive, ZipWriter};

use super::npy_header::{Header, DataType};
use super::{check_for_extra_bytes, PathOrBuffer};
use super::labels::{load_labels, save_labels};
use dlpack::sys::DLDataTypeCode;

use crate::{TensorBlock, Labels, Error, mts_array_t};


/// Check if the file/buffer in `data` looks like it could contain serialized
/// `TensorBlock`.
pub fn looks_like_block_data(mut data: PathOrBuffer) -> bool {
    match data {
        PathOrBuffer::Path(path) => {
            match std::fs::File::open(path) {
                Ok(file) => {
                    let mut buffer = BufReader::new(file);
                    return looks_like_block_data(PathOrBuffer::Buffer(&mut buffer));
                },
                Err(_) => { return false; }
            }
        },
        PathOrBuffer::Buffer(ref mut buffer) => {
            match ZipArchive::new(buffer) {
                Ok(mut archive) => {
                    return archive.by_name("values.npy").is_ok()
                }
                Err(_) => { return false; }
            }
        },
    }
}

/// Load the serialized tensor block from the given path.
///
/// Arrays for the values and gradient data will be created with the given
/// `create_array` callback, and filled by this function with the corresponding
/// data.
///
/// See the [`load`] for more information about the format used to serialize
/// `TensorBlock`.
pub fn load_block<R, F>(reader: R, create_array: F) -> Result<TensorBlock, Error>
    where R: std::io::Read + std::io::Seek,
          F: Fn(Vec<usize>) -> Result<mts_array_t, Error>
{
    let mut archive = ZipArchive::new(reader).map_err(|e| ("<root>".into(), e))?;

    return read_single_block(&mut archive, "", None, &create_array);
}

/// Save the given block to a file (or any other writer).
///
/// The format used is documented in the [`load`] function, and consists of a
/// zip archive containing NPY files. The recomended file extension when saving
/// data is `.mts`, to prevent confusion with generic `.npz` files.
pub fn save_block<W: std::io::Write + std::io::Seek>(writer: W, block: &TensorBlock) -> Result<(), Error> {
    let mut archive = ZipWriter::new(writer);
    write_single_block(&mut archive, "", true, block)?;
    archive.finish().map_err(|e| ("<root>".into(), e))?;

    return Ok(());
}


/******************************************************************************/

#[allow(clippy::needless_pass_by_value)]
pub(super) fn read_single_block<R, F>(
    archive: &mut ZipArchive<R>,
    prefix: &str,
    properties: Option<Arc<Labels>>,
    create_array: &F,
) -> Result<TensorBlock, Error>
    where R: std::io::Read + std::io::Seek,
          F: Fn(Vec<usize>) -> Result<mts_array_t, Error>
{
    let path = format!("{}values.npy", prefix);
    let data_file = archive.by_name(&path).map_err(|e| (path, e))?;
    let (data, shape) = read_data(data_file, &create_array)?;

    let path = format!("{}samples.npy", prefix);
    let samples_file = archive.by_name(&path).map_err(|e| (path, e))?;
    let samples = Arc::new(load_labels(samples_file)?);

    let mut components = Vec::new();
    for i in 0..(shape.len() - 2) {
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
        let gradient = read_single_block(
            archive,
            &format!("{}gradients/{}/", prefix, parameter),
            Some(properties.clone()),
            create_array
        )?;

        block.add_gradient(parameter, gradient)?;
    }

    return Ok(block);
}

// Read a data array from the given reader, using numpy's NPY format
fn read_data<R, F>(mut reader: R, create_array: &F) -> Result<(mts_array_t, Vec<usize>), Error>
    where R: std::io::Read, F: Fn(Vec<usize>) -> Result<mts_array_t, Error>
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

pub(super) fn write_single_block<W: std::io::Write + std::io::Seek>(
    archive: &mut ZipWriter<W>,
    prefix: &str,
    values: bool,
    block: &TensorBlock,
) -> Result<(), Error> {
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .large_file(true)
        .last_modified_time(zip::DateTime::from_date_and_time(2000, 1, 1, 0, 0, 0).expect("invalid datetime"));

    let path = format!("{}values.npy", prefix);
    archive.start_file(&path, options).map_err(|e| (path, e))?;
    write_data(archive, &block.values)?;

    let path = format!("{}samples.npy", prefix);
    archive.start_file(&path, options).map_err(|e| (path, e))?;
    save_labels(archive, &block.samples)?;

    for (i, component) in block.components.iter().enumerate() {
        let path = format!("{}components/{}.npy", prefix, i);
        archive.start_file(&path, options).map_err(|e| (path, e))?;
        save_labels(archive, component)?;
    }

    if values {
        let path = format!("{}properties.npy", prefix);
        archive.start_file(&path, options).map_err(|e| (path.clone(), e))?;
        save_labels(archive, &block.properties)?;
    }

    for (parameter, gradient) in block.gradients() {
        let prefix = format!("{}gradients/{}/", prefix, parameter);
        write_single_block(archive, &prefix, false, gradient)?;
    }

    Ok(())
}

// Write an array to the given writer, using numpy's NPY format
fn write_data<W: std::io::Write>(writer: &mut W, array: &mts_array_t) -> Result<(), Error> {
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

fn read_data_dlpack<R: std::io::Read>(
    mut reader: R,
    array: &mut mts_array_t
) -> Result<(), Error> {
    let header = Header::from_reader(&mut reader)?;
    if header.fortran_order {
        return Err(Error::Serialization("data can not be loaded from fortran-order arrays".into()));
    }

    if array.shape()? != header.shape {
        return Err(Error::Serialization(format!(
            "shape mismatch between the array and the file: got [{:?}] expected [{:?}]",
            array.shape()?, header.shape
        )));
    }

    let dl_tensor_ptr = array.as_dlpack()?;
    let _dl_tensor_deleter = scopeguard::guard(dl_tensor_ptr, |ptr| {
        if let Some(deleter) = unsafe { (*ptr).deleter } {
            unsafe { deleter(ptr) };
        }
    });

    let tensor = unsafe { &*dl_tensor_ptr };
    let len: usize = header.shape.iter().product();
    let npy_dtype = if let DataType::Scalar(s) = &header.type_descriptor {
        s.clone()
    } else {
        return Err(Error::Serialization("compound npy dtypes are not supported".into()));
    };

    unsafe {
        let code = tensor.dl_tensor.dtype.code;
        let bits = tensor.dl_tensor.dtype.bits;
        let ptr = (tensor.dl_tensor.data as *mut u8).add(tensor.dl_tensor.byte_offset as usize);

        let expected_npy_dtype = dlpack_to_npy_descr(code, bits)?;
        if !npy_dtype.ends_with(&expected_npy_dtype[1..]) {
             return Err(Error::Serialization(format!(
                "dtype mismatch: array requires '{}' but file contains '{}'",
                expected_npy_dtype, npy_dtype
            )));
        }

        let is_little_endian = npy_dtype.starts_with('<');

        match (code, bits) {
            (DLDataTypeCode::kDLFloat, 64) => {
                let slice = std::slice::from_raw_parts_mut(ptr as *mut f64, len);
                if is_little_endian {
                    reader.read_f64_into::<LittleEndian>(slice)?;
                } else {
                    reader.read_f64_into::<BigEndian>(slice)?;
                }
            },
            (DLDataTypeCode::kDLFloat, 32) => {
                let slice = std::slice::from_raw_parts_mut(ptr as *mut f32, len);
                if is_little_endian {
                    reader.read_f32_into::<LittleEndian>(slice)?;
                } else {
                    reader.read_f32_into::<BigEndian>(slice)?;
                }
            },
            (DLDataTypeCode::kDLInt, 32) => {
                let slice = std::slice::from_raw_parts_mut(ptr as *mut i32, len);
                if is_little_endian {
                    reader.read_i32_into::<LittleEndian>(slice)?;
                } else {
                    reader.read_i32_into::<BigEndian>(slice)?;
                }
            },
            _ => return Err(Error::Serialization(format!("unsupported DLPack dtype for reading: code {:?}, bits {:?}", code, bits))),
        }
    }

    check_for_extra_bytes(&mut reader)?;
    Ok(())
}


fn write_data_dlpack<W: std::io::Write>(writer: &mut W, array: &mts_array_t) -> Result<(), Error> {
    let dl_tensor_ptr = array.as_dlpack()?;
    let _dl_tensor_deleter = scopeguard::guard(dl_tensor_ptr, |ptr| {
        if let Some(deleter) = unsafe { (*ptr).deleter } {
            unsafe { deleter(ptr) };
        }
    });

    let tensor = unsafe { &*dl_tensor_ptr };
    let (code, bits) = (tensor.dl_tensor.dtype.code, tensor.dl_tensor.dtype.bits);

    let type_descriptor = dlpack_to_npy_descr(code, bits)?;

    let header = Header {
        type_descriptor: DataType::Scalar(type_descriptor.into()),
        fortran_order: false,
        shape: array.shape()?.to_vec(),
    };
    header.write(&mut *writer)?;

    unsafe {
        let len: usize = array.shape()?.iter().product();
        let ptr = (tensor.dl_tensor.data as *const u8).add(tensor.dl_tensor.byte_offset as usize);

        match (code, bits) {
            (DLDataTypeCode::kDLFloat, 64) => {
                let slice = std::slice::from_raw_parts(ptr as *const f64, len);
                for &value in slice { writer.write_f64::<NativeEndian>(value)?; }
            },
            (DLDataTypeCode::kDLFloat, 32) => {
                let slice = std::slice::from_raw_parts(ptr as *const f32, len);
                for &value in slice { writer.write_f32::<NativeEndian>(value)?; }
            },
            (DLDataTypeCode::kDLInt, 32) => {
                let slice = std::slice::from_raw_parts(ptr as *const i32, len);
                for &value in slice { writer.write_i32::<NativeEndian>(value)?; }
            },
             _ => return Err(Error::Serialization(format!("unsupported DLPack dtype for writing: code {:?}, bits {:?}", code, bits))),
        }
    }
    Ok(())
}

fn dlpack_to_npy_descr(code: DLDataTypeCode, bits: u8) -> Result<String, Error> {
    let endian = if cfg!(target_endian = "little") { "<" } else { ">" };
    
    let (type_char, type_size) = match (code, bits) {
        (DLDataTypeCode::kDLInt, 8) => ("i", 1),
        (DLDataTypeCode::kDLInt, 16) => ("i", 2),
        (DLDataTypeCode::kDLInt, 32) => ("i", 4),
        (DLDataTypeCode::kDLInt, 64) => ("i", 8),
        (DLDataTypeCode::kDLUInt, 8) => ("u", 1),
        (DLDataTypeCode::kDLUInt, 16) => ("u", 2),
        (DLDataTypeCode::kDLUInt, 32) => ("u", 4),
        (DLDataTypeCode::kDLUInt, 64) => ("u", 8),
        (DLDataTypeCode::kDLFloat, 32) => ("f", 4),
        (DLDataTypeCode::kDLFloat, 64) => ("f", 8),
        _ => return Err(Error::Serialization(format!("unsupported DLPack dtype: code {:?}, bits {:?}", code, bits))),
    };

    Ok(format!("{}{}{}", endian, type_char, type_size))
}
