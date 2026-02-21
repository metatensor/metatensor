use std::io::BufReader;
use std::collections::HashSet;
use std::sync::Arc;

use byteorder::{LittleEndian, BigEndian, NativeEndian, WriteBytesExt, ReadBytesExt};
use zip::{ZipArchive, ZipWriter};
use dlpk::sys::{DLDataTypeCode, DLDevice, DLPackVersion};

use super::npy_header::{Header, DataType};
use super::{check_for_extra_bytes, native_endian_prefix, Endianness, PathOrBuffer};
use super::labels::{load_labels, save_labels};

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

/// Parse an NPY type descriptor string (e.g. `"<f8"`) into a DLPack data type code,
/// bit width, and byte order.
pub(super) fn npy_descr_to_dtype(descr: &str) -> Result<(DLDataTypeCode, u8, Endianness), Error> {
    if descr.len() < 3 {
        return Err(Error::Serialization(format!("invalid type descriptor: {}", descr)));
    }

    let endian = match &descr[0..1] {
        "<" => Endianness::Little,
        "="  => Endianness::Native,
        ">" => Endianness::Big,
        "|" => return Err(Error::Serialization(format!(
            "endianness '|' (not applicable) is not supported in type descriptor: {}", descr
        ))),
        _ => return Err(Error::Serialization(format!("unknown endianness in type descriptor: {}", descr))),
    };

    let type_char = &descr[1..2];
    let size_str = &descr[2..];
    let size: u8 = size_str.parse().map_err(|_| {
        Error::Serialization(format!("invalid size in type descriptor: {}", descr))
    })?;

    let (code, bits) = match (type_char, size) {
        ("f", 4) => (DLDataTypeCode::kDLFloat, 32),
        ("f", 8) => (DLDataTypeCode::kDLFloat, 64),
        ("i", 1) => (DLDataTypeCode::kDLInt, 8),
        ("i", 2) => (DLDataTypeCode::kDLInt, 16),
        ("i", 4) => (DLDataTypeCode::kDLInt, 32),
        ("i", 8) => (DLDataTypeCode::kDLInt, 64),
        ("u", 1) => (DLDataTypeCode::kDLUInt, 8),
        ("u", 2) => (DLDataTypeCode::kDLUInt, 16),
        ("u", 4) => (DLDataTypeCode::kDLUInt, 32),
        ("u", 8) => (DLDataTypeCode::kDLUInt, 64),
        ("b", 1) => (DLDataTypeCode::kDLBool, 8),
        ("c", 8) => (DLDataTypeCode::kDLComplex, 64),
        ("c", 16) => (DLDataTypeCode::kDLComplex, 128),
        ("f", 2) => (DLDataTypeCode::kDLFloat, 16),
        _ => return Err(Error::Serialization(format!("unsupported type descriptor: {}", descr))),
    };

    Ok((code, bits, endian))
}


fn read_as<T, R>(reader: &mut R, tensor: dlpk::DLPackTensorRefMut<'_>, cb: impl Fn(&mut R, &mut T) -> Result<(), std::io::Error>) -> Result<(), Error>
where R: std::io::Read,
      T: dlpk::DLPackPointerCast + 'static
{
    let mut view: ndarray::ArrayViewMutD<T> = tensor.try_into()
        .map_err(|e| Error::Serialization(format!("failed to convert DLPack to ndarray mutable view: {}", e)))?;

    for value in &mut view {
        cb(reader, value)?;
    }

    Ok(())
}

// Read a data array from the given reader, using numpy's NPY format
#[allow(clippy::too_many_lines)]
fn read_data<R, F>(mut reader: R, create_array: &F) -> Result<(mts_array_t, Vec<usize>), Error>
    where R: std::io::Read, F: Fn(Vec<usize>) -> Result<mts_array_t, Error>
{
    let header = Header::from_reader(&mut reader)?;
    if header.fortran_order {
        return Err(Error::Serialization("data can not be loaded from fortran-order arrays".into()));
    }

    let shape = header.shape;
    let array = create_array(shape.clone())?;

    let num_elements: usize = shape.iter().product();
    if num_elements == 0 {
        check_for_extra_bytes(&mut reader)?;
        return Ok((array, shape));
    }

    let device = DLDevice::cpu();
    let version = DLPackVersion::current();
    let mut dl_tensor = array.as_dlpack(device, None, version)?;

    let descr = if let DataType::Scalar(s) = &header.type_descriptor {
        s.as_str()
    } else {
        return Err(Error::Serialization("structured arrays are not supported".into()));
    };

    let (file_code, file_bits, endian) = npy_descr_to_dtype(descr)?;

    let tensor_ref = dl_tensor.as_mut();

    // Endianness is handled inside each arm to avoid tripling the number of
    // match arms (which inflates uncovered-line counts for big/native paths
    // that are not exercised in tests on little-endian CI).
    match (file_code, file_bits) {
        // Standard Floats
        (DLDataTypeCode::kDLFloat, 32) => read_as::<f32, _>(&mut reader, tensor_ref, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_f32::<LittleEndian>()?,
                Endianness::Big => r.read_f32::<BigEndian>()?,
                Endianness::Native => r.read_f32::<NativeEndian>()?,
            };
            Ok(())
        }),
        (DLDataTypeCode::kDLFloat, 64) => read_as::<f64, _>(&mut reader, tensor_ref, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_f64::<LittleEndian>()?,
                Endianness::Big => r.read_f64::<BigEndian>()?,
                Endianness::Native => r.read_f64::<NativeEndian>()?,
            };
            Ok(())
        }),

        // Standard Ints
        (DLDataTypeCode::kDLInt, 8) => read_as::<i8, _>(&mut reader, tensor_ref, |r: &mut R, v| {
            *v = r.read_i8()?;
            Ok(())
        }),
        (DLDataTypeCode::kDLInt, 16) => read_as::<i16, _>(&mut reader, tensor_ref, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_i16::<LittleEndian>()?,
                Endianness::Big => r.read_i16::<BigEndian>()?,
                Endianness::Native => r.read_i16::<NativeEndian>()?,
            };
            Ok(())
        }),
        (DLDataTypeCode::kDLInt, 32) => read_as::<i32, _>(&mut reader, tensor_ref, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_i32::<LittleEndian>()?,
                Endianness::Big => r.read_i32::<BigEndian>()?,
                Endianness::Native => r.read_i32::<NativeEndian>()?,
            };
            Ok(())
        }),
        (DLDataTypeCode::kDLInt, 64) => read_as::<i64, _>(&mut reader, tensor_ref, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_i64::<LittleEndian>()?,
                Endianness::Big => r.read_i64::<BigEndian>()?,
                Endianness::Native => r.read_i64::<NativeEndian>()?,
            };
            Ok(())
        }),

        // Unsigned Ints
        (DLDataTypeCode::kDLUInt, 8) => read_as::<u8, _>(&mut reader, tensor_ref, |r: &mut R, v| {
            *v = r.read_u8()?;
            Ok(())
        }),
        (DLDataTypeCode::kDLUInt, 16) => read_as::<u16, _>(&mut reader, tensor_ref, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_u16::<LittleEndian>()?,
                Endianness::Big => r.read_u16::<BigEndian>()?,
                Endianness::Native => r.read_u16::<NativeEndian>()?,
            };
            Ok(())
        }),
        (DLDataTypeCode::kDLUInt, 32) => read_as::<u32, _>(&mut reader, tensor_ref, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_u32::<LittleEndian>()?,
                Endianness::Big => r.read_u32::<BigEndian>()?,
                Endianness::Native => r.read_u32::<NativeEndian>()?,
            };
            Ok(())
        }),
        (DLDataTypeCode::kDLUInt, 64) => read_as::<u64, _>(&mut reader, tensor_ref, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => r.read_u64::<LittleEndian>()?,
                Endianness::Big => r.read_u64::<BigEndian>()?,
                Endianness::Native => r.read_u64::<NativeEndian>()?,
            };
            Ok(())
        }),

        // Boolean (Read as u8)
        (DLDataTypeCode::kDLBool, 8) => read_as::<bool, _>(&mut reader, tensor_ref, |r: &mut R, v| {
            *v = r.read_u8()? != 0;
            Ok(())
        }),

        // Complex Numbers (Read as array of 2 floats)
        (DLDataTypeCode::kDLComplex, 64) => read_as::<[f32; 2], _>(&mut reader, tensor_ref, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => [r.read_f32::<LittleEndian>()?, r.read_f32::<LittleEndian>()?],
                Endianness::Big => [r.read_f32::<BigEndian>()?, r.read_f32::<BigEndian>()?],
                Endianness::Native => [r.read_f32::<NativeEndian>()?, r.read_f32::<NativeEndian>()?],
            };
            Ok(())
        }),
        (DLDataTypeCode::kDLComplex, 128) => read_as::<[f64; 2], _>(&mut reader, tensor_ref, |r: &mut R, v| {
            *v = match endian {
                Endianness::Little => [r.read_f64::<LittleEndian>()?, r.read_f64::<LittleEndian>()?],
                Endianness::Big => [r.read_f64::<BigEndian>()?, r.read_f64::<BigEndian>()?],
                Endianness::Native => [r.read_f64::<NativeEndian>()?, r.read_f64::<NativeEndian>()?],
            };
            Ok(())
        }),

        // Float16 (Read as half::f16)
        (DLDataTypeCode::kDLFloat, 16) => read_as::<half::f16, _>(&mut reader, tensor_ref, |r: &mut R, v| {
            *v = half::f16::from_bits(match endian {
                Endianness::Little => r.read_u16::<LittleEndian>()?,
                Endianness::Big => r.read_u16::<BigEndian>()?,
                Endianness::Native => r.read_u16::<NativeEndian>()?,
            });
            Ok(())
        }),

        _ => Err(Error::Serialization(format!(
            "unsupported dtype for reading: {:?} {} bits", file_code, file_bits
        ))),
    }?;

    check_for_extra_bytes(&mut reader)?;
    Ok((array, shape))
}

pub(super) fn write_single_block<W: std::io::Write + std::io::Seek>(
    archive: &mut ZipWriter<W>,
    prefix: &str,
    values: bool,
    block: &TensorBlock,
) -> Result<(), Error> {
    let options = zip::write::FileOptions::default()
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

fn dlpack_to_npy_descr(code: DLDataTypeCode, bits: u8) -> Result<String, Error> {
    let endian = native_endian_prefix();

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
        (DLDataTypeCode::kDLBool, 8) => ("b", 1),
        (DLDataTypeCode::kDLComplex, 64) => ("c", 8),
        (DLDataTypeCode::kDLComplex, 128) => ("c", 16),
        (DLDataTypeCode::kDLFloat, 16) => ("f", 2),
        _ => return Err(Error::Serialization(
            format!("unsupported DLPack dtype: code {:?}, bits {:?}", code, bits)
                                            )
        ),
    };

    Ok(format!("{}{}{}", endian, type_char, type_size))
}


fn write_as<T, W>(writer: &mut W, tensor: dlpk::DLPackTensorRef<'_>, cb: impl Fn(&mut W, T) -> Result<(), std::io::Error>) -> Result<(), Error>
where W: std::io::Write,
      T: Copy + dlpk::DLPackPointerCast + 'static
{
    let view: ndarray::ArrayViewD<T> = tensor.try_into()
        .map_err(|e| Error::Serialization(format!("failed to convert DLPack to ndarray view: {}", e)))?;

    for &value in &view {
        cb(writer, value)?;
    }

    Ok(())
}

// Write an array to the given writer, using numpy's NPY format
fn write_data<W: std::io::Write>(writer: &mut W, array: &mts_array_t) -> Result<(), Error> {
    let device = DLDevice::cpu();
    let version = DLPackVersion::current();

    // Get DLPack Tensor
    let dl_tensor = array.as_dlpack(device, None, version)?;
    let tensor_ref = dl_tensor.as_ref();
    let dtype = tensor_ref.raw.dtype;
    let (code, bits) = (dtype.code, dtype.bits);

    // Validate Lanes
    if dtype.lanes != 1 {
        return Err(Error::Serialization(format!(
            "unsupported DLPack dtype: lanes != 1 ({})", dtype.lanes
        )));
    }

    // Write Header
    let tdesc = super::block::dlpack_to_npy_descr(code, bits)?;
    let header = Header {
        type_descriptor: DataType::Scalar(tdesc),
        fortran_order: false,
        shape: array.shape()?.to_vec(),
    };

    header.write(&mut *writer)?;

    // Get metadata for size and pointer for data
    let num_elements: usize = header.shape.iter().product();
    if num_elements == 0 {
        return Ok(());
    }

    match (code, bits) {
        // Standard Floats
        (DLDataTypeCode::kDLFloat, 32) => write_as::<f32, _>(writer, tensor_ref, |w: &mut W, v| w.write_f32::<NativeEndian>(v)),
        (DLDataTypeCode::kDLFloat, 64) => write_as::<f64, _>(writer, tensor_ref, |w: &mut W, v| w.write_f64::<NativeEndian>(v)),

        // Standard Ints
        (DLDataTypeCode::kDLInt, 8) => write_as::<i8, _>(writer, tensor_ref, |w: &mut W, v| w.write_i8(v)),
        (DLDataTypeCode::kDLInt, 16) => write_as::<i16, _>(writer, tensor_ref, |w: &mut W, v| w.write_i16::<NativeEndian>(v)),
        (DLDataTypeCode::kDLInt, 32) => write_as::<i32, _>(writer, tensor_ref, |w: &mut W, v| w.write_i32::<NativeEndian>(v)),
        (DLDataTypeCode::kDLInt, 64) => write_as::<i64, _>(writer, tensor_ref, |w: &mut W, v| w.write_i64::<NativeEndian>(v)),

        // Unsigned Ints
        (DLDataTypeCode::kDLUInt, 8) => write_as::<u8, _>(writer, tensor_ref, |w: &mut W, v| w.write_u8(v)),
        (DLDataTypeCode::kDLUInt, 16) => write_as::<u16, _>(writer, tensor_ref, |w: &mut W, v| w.write_u16::<NativeEndian>(v)),
        (DLDataTypeCode::kDLUInt, 32) => write_as::<u32, _>(writer, tensor_ref, |w: &mut W, v| w.write_u32::<NativeEndian>(v)),
        (DLDataTypeCode::kDLUInt, 64) => write_as::<u64, _>(writer, tensor_ref, |w: &mut W, v| w.write_u64::<NativeEndian>(v)),

        // Boolean, stored as u8
        (DLDataTypeCode::kDLBool, 8) => write_as::<bool, _>(writer, tensor_ref, |w: &mut W, v| w.write_u8(u8::from(v))),
        // Float16, stored as u16 via half::f16
        (DLDataTypeCode::kDLFloat, 16) => write_as::<half::f16, _>(writer, tensor_ref, |w: &mut W, v| w.write_u16::<NativeEndian>(v.to_bits())),

        // Complex Numbers
        (DLDataTypeCode::kDLComplex, 64) => write_as::<[f32; 2], _>(writer, tensor_ref, |w: &mut W, v| {
            w.write_f32::<NativeEndian>(v[0])?;
            w.write_f32::<NativeEndian>(v[1])
        }),
        (DLDataTypeCode::kDLComplex, 128) => write_as::<[f64; 2], _>(writer, tensor_ref, |w: &mut W, v| {
            w.write_f64::<NativeEndian>(v[0])?;
            w.write_f64::<NativeEndian>(v[1])
        }),

        _ => Err(Error::Serialization(format!("unsupported dtype for writing: {:?} {} bits", code, bits))),
    }
}
