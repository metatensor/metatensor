use byteorder::{LittleEndian, ReadBytesExt, BigEndian, WriteBytesExt, NativeEndian};

use super::{Header, DataType, check_for_extra_bytes};
use crate::{Error, Labels, LabelsBuilder, LabelValue};

/// Read `Labels` stored using numpy's NPY format.
///
/// The labels are stored as a structured array, as if each entry in the Labels
/// was a C struct of 32-bit integers. The corresponding `dtype` is a list
/// associating name and either "<i4" for little endian file or ">i4" for big
/// endian file. Data is stored in exactly the same way as inside a `Labels`,
/// i.e. a big blob of 32-bit integers.
pub fn read_npy_labels<R: std::io::Read>(mut reader: R) -> Result<Labels, Error> {
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

    let mut builder = LabelsBuilder::new(names.iter().map(|s| &**s).collect());
    for chunk in data.chunks_exact(names.len()) {
        builder.add(&chunk.iter().map(|&i| LabelValue::new(i)).collect::<Vec<_>>())?;
    }

    return Ok(builder.finish());
}

/// Write `Labels` to the writer using numpy's NPY format.
///
/// See [`read_npy_labels`] for more information on how `Labels` are stored to
/// files.
pub fn write_npy_labels<W: std::io::Write>(writer: &mut W, labels: &Labels) -> Result<(), Error> {
    let mut type_descriptor = Vec::new();
    for name in labels.names() {
        if cfg!(target_endian = "little") {
            type_descriptor.push((name.into(), "<i4".into()));
        } else {
            assert!(cfg!(target_endian = "big"));
            type_descriptor.push((name.into(), ">i4".into()));
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
fn check_type_descriptor(desc: &DataType) -> Result<(Vec<String>, Endianness), Error> {
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

                names.push(name.clone());
            }
        },
        DataType::Scalar(_) => {
            return Err(error);
        }
    }

    return Ok((names, endianness.expect("failed to find endianness")));
}
