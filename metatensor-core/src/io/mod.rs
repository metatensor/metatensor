mod npy_header;

mod labels;
pub use self::labels::load_labels;
pub use self::labels::save_labels;
pub use self::labels::looks_like_labels_data;

mod block;
pub use self::block::load_block;
pub use self::block::save_block;
pub use self::block::looks_like_block_data;

mod tensor;
pub use self::tensor::load;
pub use self::tensor::save;
pub use self::tensor::looks_like_tensormap_data;

mod mmap;
pub use self::mmap::{load_mmap, load_block_mmap};


use crate::Error;

pub trait ReadAndSeek: std::io::Read + std::io::Seek {}
impl<T: std::io::Read + std::io::Seek> ReadAndSeek for T {}

pub enum PathOrBuffer<'a> {
    Path(&'a str),
    Buffer(&'a mut dyn ReadAndSeek),
}

/// Byte order for multi-byte values in NPY files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Endianness {
    Little,
    Big,
    Native,
}

/// Return the NPY endianness prefix character for the native byte order.
pub(crate) fn native_endian_prefix() -> &'static str {
    if cfg!(target_endian = "little") { "<" } else { ">" }
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


/// Parse a STORED (uncompressed) NPY entry inside a memory-mapped ZIP
/// archive without copying the data. Returns the array shape, DLPack
/// dtype, and byte offset of the raw element data within the
/// mmap-ed file. Both mmap and partial-load code paths use this --
/// the on-disk format checks (STORED compression, native byte order,
/// non-fortran order, non-structured dtype) are identical for both.
pub(crate) fn parse_stored_npy_entry(
    archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
    mmap: &[u8],
    path: &str,
) -> Result<(Vec<usize>, dlpk::sys::DLDataType, usize), Error> {
    use std::io::Cursor;
    use crate::io::npy_header::{DataType, Header};
    use crate::io::block::npy_descr_to_dtype;

    let entry = archive.by_name(path).map_err(|e| (path.to_string(), e))?;

    if entry.compression() != zip::CompressionMethod::Stored {
        return Err(Error::Serialization(format!(
            "entry '{}' uses compression {:?}, but zero-copy NPY parsing requires STORED entries",
            path, entry.compression()
        )));
    }

    let entry_size = entry.size() as usize;
    let data_start = entry.data_start() as usize;
    drop(entry);

    let end = data_start.checked_add(entry_size).ok_or_else(|| {
        Error::Serialization(format!("entry '{}' has overflowing size+offset", path))
    })?;
    if end > mmap.len() {
        return Err(Error::Serialization(format!(
            "entry '{}' extends beyond the end of the file",
            path
        )));
    }

    let npy_bytes = &mmap[data_start..end];
    let mut cursor = Cursor::new(npy_bytes);
    let header = Header::from_reader(&mut cursor).map_err(|e| {
        Error::Serialization(format!("invalid NPY header in '{}': {}", path, e))
    })?;

    if header.fortran_order {
        return Err(Error::Serialization(format!(
            "fortran-order arrays are not supported (in '{}')",
            path
        )));
    }

    let descr = match &header.type_descriptor {
        DataType::Scalar(s) => s.as_str(),
        _ => {
            return Err(Error::Serialization(format!(
                "structured arrays are not supported (in '{}')",
                path
            )));
        }
    };

    let (code, bits, endian) = npy_descr_to_dtype(descr)?;

    let native_ok = match endian {
        Endianness::Native => true,
        Endianness::Little => cfg!(target_endian = "little"),
        Endianness::Big => cfg!(target_endian = "big"),
    };
    if !native_ok {
        return Err(Error::Serialization(format!(
            "zero-copy NPY parsing requires native byte order, but entry '{}' uses '{}'",
            path, descr
        )));
    }

    let dl_dtype = dlpk::sys::DLDataType { code, bits, lanes: 1 };
    let header_len = cursor.position() as usize;
    let raw_data_offset = data_start + header_len;

    Ok((header.shape, dl_dtype, raw_data_offset))
}


/// Enumerate the gradient parameter names under a given prefix in a
/// single pass through the archive's filename list. Used by mmap.rs
/// and partial.rs.
pub(crate) fn discover_gradient_parameters<R: std::io::Read + std::io::Seek>(
    archive: &zip::ZipArchive<R>,
    prefix: &str,
) -> std::collections::HashSet<String> {
    let mut parameters = std::collections::HashSet::new();
    let gradient_prefix = format!("{}gradients/", prefix);
    for name in archive.file_names() {
        if name.starts_with(&gradient_prefix) && name.ends_with("/samples.npy") {
            let (_, parameter) = name.split_at(gradient_prefix.len());
            let parameter = parameter.split('/').next().expect("gradient parameter");
            parameters.insert(parameter.to_string());
        }
    }
    parameters
}


/// Load `info.json` from a ZIP archive (if present) and call
/// `add_info(key, value)` for each entry. Used by `tensor.rs::load`,
/// `mmap.rs::load_mmap`, and `partial.rs::load_partial`.
pub(crate) fn load_info_json<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    mut add_info: impl FnMut(&str, &str),
) -> Result<(), Error> {
    use std::io::Read;
    let info_path = String::from("info.json");
    if !archive.file_names().any(|name| name == info_path) {
        return Ok(());
    }
    let mut info_file = archive.by_name(&info_path).map_err(|e| (info_path, e))?;
    let mut info = String::new();
    info_file.read_to_string(&mut info)?;
    let info = jzon::parse(&info).map_err(|e| Error::Serialization(e.to_string()))?;
    let info = info
        .as_object()
        .ok_or_else(|| Error::Serialization("'info.json' should contain an object".into()))?;

    for (key, value) in info.iter() {
        let value = value
            .as_str()
            .ok_or_else(|| Error::Serialization("values in 'info.json' should be strings".into()))?;
        add_info(key, value);
    }
    Ok(())
}


