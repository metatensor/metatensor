mod npy_header;
mod labels;

mod load;
pub use self::load::load;

mod save;
pub use self::save::save;

use crate::Error;

// returns an error if the given reader contains any more data
fn check_for_extra_bytes<R: std::io::Read>(reader: &mut R) -> Result<(), Error> {
    let extra = reader.read_to_end(&mut Vec::new())?;
    if extra == 0 {
        Ok(())
    } else {
        Err(Error::Serialization(format!("found {} extra bytes after the expected end of data", extra)))
    }
}
