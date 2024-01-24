mod npy_header;
mod labels;
pub use self::labels::load_labels;
pub use self::labels::looks_like_labels_data;

mod load;
pub use self::load::load;
pub use self::load::looks_like_tensormap_data;

mod save;
pub use self::save::save;
pub use self::labels::save_labels;

use crate::Error;

pub trait ReadAndSeek: std::io::Read + std::io::Seek {}
impl<T: std::io::Read + std::io::Seek> ReadAndSeek for T {}

pub enum PathOrBuffer<'a> {
    Path(&'a str),
    Buffer(&'a mut dyn ReadAndSeek),
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
