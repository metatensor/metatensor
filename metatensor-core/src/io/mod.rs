mod npy_header;

mod mmap_array;
mod mmap;
pub use self::mmap::{load_mmap, load_block_mmap, load_mmap_with, load_block_mmap_with};

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
