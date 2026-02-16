use std::io::BufReader;

use byteorder::{LittleEndian, ReadBytesExt, BigEndian, WriteBytesExt, NativeEndian};

use super::npy_header::{Header, DataType};
use super::{check_for_extra_bytes, PathOrBuffer};
use crate::labels::LabelValue;
use crate::{Error, Labels};


/// Check if the file/buffer in `data` looks like it could contain serialized
/// `Labels`.
///
/// We check the file extension and if the file content starts with the right
/// header for NPY files.
pub fn looks_like_labels_data(mut data: PathOrBuffer) -> bool {
    match data {
        PathOrBuffer::Path(path) => {
            let path = std::path::Path::new(path);
            if let Some(extension) = path.extension() {
                if extension.eq_ignore_ascii_case("npy") {
                    return true;
                }
            }

            match std::fs::File::open(path) {
                Ok(file) => {
                    let mut buffer = BufReader::new(file);
                    return looks_like_labels_data(PathOrBuffer::Buffer(&mut buffer));
                },
                Err(_) => { return false; }
            }
        },
        PathOrBuffer::Buffer(ref mut buffer) => {
            return Header::from_reader(buffer).is_ok();
        },
    }
}

/// Read `Labels` stored using numpy's NPY format.
///
/// The labels are stored as a structured array, as if each entry in the Labels
/// was a C struct of 32-bit integers. The corresponding `dtype` is a list
/// associating name and either "<i4" for little endian file or ">i4" for big
/// endian file. Data is stored in exactly the same way as inside a `Labels`,
/// i.e. a big blob of 32-bit integers.
pub fn load_labels<R: std::io::Read>(mut reader: R) -> Result<Labels, Error> {
    let header = Header::from_reader(&mut reader)?;
    if header.fortran_order {
        return Err(Error::Serialization("Labels can not be loaded from fortran-order arrays".into()));
    } else if header.shape.len() != 1 {
        return Err(Error::Serialization("Expected a 1-D array when loading Labels".into()));
    }
    let (names, endianness) = check_type_descriptor(&header.type_descriptor)?;

    let mut data = vec![0; header.shape[0] * names.len()];
    match endianness {
        Endianness::LittleEndian => reader.read_i32_into::<LittleEndian>(&mut data)?,
        Endianness::BigEndian => reader.read_i32_into::<BigEndian>(&mut data)?,
    }

    check_for_extra_bytes(&mut reader)?;

    let values = unsafe {
        // This is the recomended version of `std::mem::transmute` for Vec. It
        // is safe because `LabelValue` is a #[repr(transparent)] wrapper for
        // i32
        let mut data = std::mem::ManuallyDrop::new(data);
        Vec::from_raw_parts(
            data.as_mut_ptr().cast::<LabelValue>(),
            data.len(),
            data.capacity()
        )
    };

    return Labels::new(&names, values);
}

/// Write `Labels` to the writer using numpy's NPY format.
///
/// See [`read_npy_labels`] for more information on how `Labels` are stored to
/// files. The recomended file extension when saving data is `.mts`, to prevent
/// confusion with generic `.npz` files.
pub fn save_labels<W: std::io::Write>(writer: &mut W, labels: &Labels) -> Result<(), Error> {
    let mut type_descriptor = Vec::new();
    for name in labels.names() {
        if cfg!(target_endian = "little") {
            type_descriptor.push((name.into(), "<i4".into()));
        } else if cfg!(target_endian = "big") {
            type_descriptor.push((name.into(), ">i4".into()));
        } else {
            unreachable!("unknown target endianness");
        }
    }

    let header = Header {
        type_descriptor: DataType::Compound(type_descriptor),
        fortran_order: false,
        shape: vec![labels.count()],
    };
    header.write(&mut *writer)?;

    for entry in labels {
        for value in entry {
            writer.write_i32::<NativeEndian>(value.i32())?;
        }
    }

    return Ok(());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endianness {
    BigEndian,
    LittleEndian,
}

/// Check that the given type descriptor matches the expected one for Labels and
/// return the corresponding set of names & endianness.
fn check_type_descriptor(desc: &DataType) -> Result<(Vec<&str>, Endianness), Error> {
    let mut names = Vec::new();

    let error = Error::Serialization("invalid dtype for labels".into());

    let mut endianness = None;
    match desc {
        DataType::Compound(list) => {
            for (name, typ) in list {
                if endianness.is_none() {
                    if typ == "<i4" {
                        endianness = Some(Endianness::LittleEndian);
                    } else if typ == ">i4" {
                        endianness = Some(Endianness::BigEndian);
                    }
                }

                if typ == "<i4" {
                    if endianness != Some(Endianness::LittleEndian) {
                        return Err(error);
                    }
                } else if typ == ">i4" {
                    if endianness != Some(Endianness::BigEndian) {
                        return Err(error);
                    }
                } else {
                    return Err(error);
                }

                names.push(&**name);
            }
        },
        DataType::Scalar(_) => {
            return Err(error);
        }
    }

    return Ok((names, endianness.expect("failed to find endianness")));
}
