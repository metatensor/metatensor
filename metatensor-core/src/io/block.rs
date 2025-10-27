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
    let (data, shape) = read_data(data_file, create_array)?;

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

    let shape = header.shape.clone();
    let (array, shape) = read_data_dlpack(reader, create_array, shape)?;

    Ok((array, shape))
}

/// Read a data array from the given reader using DLPack.
fn read_data_dlpack<R, F>(
    mut reader: R,
    create_array: &F,
    shape: Vec<usize>,
) -> Result<(mts_array_t, Vec<usize>), Error>
    where R: std::io::Read, F: Fn(Vec<usize>) -> Result<mts_array_t, Error>
{
    let array = create_array(shape.clone())?;

    if array.shape()? != shape {
        return Err(Error::Serialization(format!(
            "shape mismatch between the array and the file: got [{:?}] expected [{:?}]",
            array.shape()?, shape
        )));
    }

    let dl_tensor = array.as_dlpack()?;
    let tensor_ref = dl_tensor.as_ref();

    let len: usize = shape.iter().product();

    unsafe {
        let dtype = tensor_ref.raw.dtype;
        let code = dtype.code;
        let bits = dtype.bits;
        let ptr = (tensor_ref.raw.data as *mut u8).add(tensor_ref.byte_offset());

        match (code, bits) {
            (DLDataTypeCode::kDLFloat, 64) => {
                let slice = std::slice::from_raw_parts_mut(ptr as *mut f64, len);
                reader.read_f64_into::<LittleEndian>(slice)?;
            },
            (DLDataTypeCode::kDLFloat, 32) => {
                let slice = std::slice::from_raw_parts_mut(ptr as *mut f32, len);
                reader.read_f32_into::<LittleEndian>(slice)?;
            },
            (DLDataTypeCode::kDLInt, 32) => {
                let slice = std::slice::from_raw_parts_mut(ptr as *mut i32, len);
                reader.read_i32_into::<LittleEndian>(slice)?;
            },
            _ => return Err(Error::Serialization(format!("unsupported DLPack dtype for reading: code {:?}, bits {:?}", code, bits))),
        }
    }

    check_for_extra_bytes(&mut reader)?;

    Ok((array, shape))
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
    let dl_tensor = array.as_dlpack()?;
    let tensor_ref = dl_tensor.as_ref();
    let dtype = tensor_ref.raw.dtype;
    let (code, bits) = (dtype.code, dtype.bits);

    let type_descriptor = dlpack_to_npy_descr(code, bits)?;

    let header = Header {
        type_descriptor: DataType::Scalar(type_descriptor.into()),
        fortran_order: false,
        shape: array.shape()?.to_vec(),
    };
    header.write(&mut *writer)?;

    unsafe {
        let len: usize = array.shape()?.iter().product();
        let ptr = (tensor_ref.raw.data as *const u8)
            .add(tensor_ref.byte_offset());

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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::TestArray;
    use std::io::Cursor;

    macro_rules! test_io {
        ($name:ident, $ty:ty) => {
            #[test]
            fn $name() {
                let shape = vec![2, 3];
                let (array, initial_data) = TestArray::new_typed::<$ty>(shape.clone(), "rust.TestArray");

                let mut buffer = Vec::new();
                write_data(&mut Cursor::new(&mut buffer), &array).unwrap();

                // create a new empty array and read the data into it
                let (read_array, _read_shape) = read_data(Cursor::new(&buffer), &|shape| Ok(TestArray::new_typed::<$ty>(shape, "rust.TestArray").0)).unwrap();

                // Verify the data using DLPackTensor
                let dl1 = array.as_dlpack().unwrap();
                let dl2 = read_array.as_dlpack().unwrap();

                let len: usize = shape.iter().product();
                unsafe {
                    let data1_ptr = dl1.data_ptr::<$ty>().unwrap();
                    let data2_ptr = dl2.data_ptr::<$ty>().unwrap();
                    
                    let data1 = std::slice::from_raw_parts(data1_ptr, len);
                    let data2 = std::slice::from_raw_parts(data2_ptr, len);

                    // round-trip
                    assert_eq!(data1, &initial_data);
                    assert_eq!(data2, &initial_data);
                }
                
            }
        };
    }

    test_io!(dlpack_io_f64, f64);
    test_io!(dlpack_io_f32, f32);
    test_io!(dlpack_io_i32, i32);
}
