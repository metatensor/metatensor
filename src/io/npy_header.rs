// This file was taken from https://github.com/jturner314/ndarray-npy, version
// 0.8.1. It is Copyright 2018–2021 Jim Turner and ndarray-npy developers,
// released under MIT and Apache Licenses.

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use num_traits::ToPrimitive;
use py_literal::{
    FormatError as PyValueFormatError, ParseError as PyValueParseError, Value as PyValue,
};
use std::convert::TryFrom;
use std::error::Error;

/// Magic string to indicate npy format.
const MAGIC_STRING: &[u8] = b"\x93NUMPY";

/// The total header length (including magic string, version number, header
/// length value, array format description, padding, and final newline) must be
/// evenly divisible by this value.
// If this changes, update the docs of `ViewNpyExt` and `ViewMutNpyExt`.
const HEADER_DIVISOR: usize = 64;

#[derive(Debug)]
pub enum ParseHeaderError {
    MagicString,
    Version {
        major: u8,
        minor: u8,
    },
    /// Indicates that the `HEADER_LEN` doesn't fit in `usize`.
    HeaderLengthOverflow(u32),
    /// Indicates that the array format string contains non-ASCII characters.
    /// This is an error for .npy format versions 1.0 and 2.0.
    NonAscii,
    /// Error parsing the array format string as UTF-8. This does not apply to
    /// .npy format versions 1.0 and 2.0, which require the array format string
    /// to be ASCII.
    Utf8Parse(std::str::Utf8Error),
    UnknownKey(PyValue),
    MissingKey(String),
    IllegalValue {
        key: String,
        value: PyValue,
    },
    DictParse(PyValueParseError),
    MetaNotDict(PyValue),
    MissingNewline,
}

impl Error for ParseHeaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ParseHeaderError::Utf8Parse(err) => Some(err),
            ParseHeaderError::DictParse(err) => Some(err),
            ParseHeaderError::MagicString |
            ParseHeaderError::Version { .. } |
            ParseHeaderError::HeaderLengthOverflow(_) |
            ParseHeaderError::NonAscii |
            ParseHeaderError::UnknownKey(_) |
            ParseHeaderError::MissingKey(_) |
            ParseHeaderError::IllegalValue { .. } |
            ParseHeaderError::MetaNotDict(_) |
            ParseHeaderError::MissingNewline => None,
        }
    }
}

impl std::fmt::Display for ParseHeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ParseHeaderError::MagicString => write!(f, "start does not match magic string"),
            ParseHeaderError::Version { major, minor } => write!(f, "unknown version number: {}.{}", major, minor),
            ParseHeaderError::HeaderLengthOverflow(header_len) => write!(f, "HEADER_LEN {} does not fit in `usize`", header_len),
            ParseHeaderError::NonAscii => write!(f, "non-ascii in array format string; this is not supported in .npy format versions 1.0 and 2.0"),
            ParseHeaderError::Utf8Parse(err) => write!(f, "error parsing array format string as UTF-8: {}", err),
            ParseHeaderError::UnknownKey(key) => write!(f, "unknown key: {}", key),
            ParseHeaderError::MissingKey(key) => write!(f, "missing key: {}", key),
            ParseHeaderError::IllegalValue { key, value } => write!(f, "illegal value for key {}: {}", key, value),
            ParseHeaderError::DictParse(err) => write!(f, "error parsing metadata dict: {}", err),
            ParseHeaderError::MetaNotDict(value) => write!(f, "metadata is not a dict: {}", value),
            ParseHeaderError::MissingNewline => write!(f, "newline missing at end of header"),
        }
    }
}

impl From<std::str::Utf8Error> for ParseHeaderError {
    fn from(err: std::str::Utf8Error) -> ParseHeaderError {
        ParseHeaderError::Utf8Parse(err)
    }
}

impl From<PyValueParseError> for ParseHeaderError {
    fn from(err: PyValueParseError) -> ParseHeaderError {
        ParseHeaderError::DictParse(err)
    }
}

#[derive(Debug)]
pub enum ReadHeaderError {
    Io(std::io::Error),
    Parse(ParseHeaderError),
}

impl Error for ReadHeaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ReadHeaderError::Io(err) => Some(err),
            ReadHeaderError::Parse(err) => Some(err),
        }
    }
}

impl std::fmt::Display for ReadHeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ReadHeaderError::Io(err) => write!(f, "I/O error: {}", err),
            ReadHeaderError::Parse(err) => write!(f, "error parsing header: {}", err),
        }
    }
}

impl From<std::io::Error> for ReadHeaderError {
    fn from(err: std::io::Error) -> ReadHeaderError {
        ReadHeaderError::Io(err)
    }
}

impl From<ParseHeaderError> for ReadHeaderError {
    fn from(err: ParseHeaderError) -> ReadHeaderError {
        ReadHeaderError::Parse(err)
    }
}

#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
enum Version {
    V1_0,
    V2_0,
    V3_0,
}

impl Version {
    /// Number of bytes taken up by version number (1 byte for major version, 1
    /// byte for minor version).
    const VERSION_NUM_BYTES: usize = 2;

    fn from_bytes(bytes: &[u8]) -> Result<Self, ParseHeaderError> {
        debug_assert_eq!(bytes.len(), Self::VERSION_NUM_BYTES);
        match (bytes[0], bytes[1]) {
            (0x01, 0x00) => Ok(Version::V1_0),
            (0x02, 0x00) => Ok(Version::V2_0),
            (0x03, 0x00) => Ok(Version::V3_0),
            (major, minor) => Err(ParseHeaderError::Version { major, minor }),
        }
    }

    /// Major version number.
    fn major_version(self) -> u8 {
        match self {
            Version::V1_0 => 1,
            Version::V2_0 => 2,
            Version::V3_0 => 3,
        }
    }

    /// Major version number.
    fn minor_version(self) -> u8 {
        match self {
            Version::V1_0 | Version::V2_0 | Version::V3_0 => 0,
        }
    }

    /// Number of bytes in representation of header length.
    fn header_len_num_bytes(self) -> usize {
        match self {
            Version::V1_0 => 2,
            Version::V2_0 | Version::V3_0 => 4,
        }
    }

    /// Read header length.
    fn read_header_len<R: std::io::Read>(self, reader: &mut R) -> Result<usize, ReadHeaderError> {
        match self {
            Version::V1_0 => Ok(usize::from(reader.read_u16::<LittleEndian>()?)),
            Version::V2_0 | Version::V3_0 => {
                let header_len: u32 = reader.read_u32::<LittleEndian>()?;
                Ok(usize::try_from(header_len)
                    .map_err(|_| ParseHeaderError::HeaderLengthOverflow(header_len))?)
            }
        }
    }

    /// Format header length as bytes for writing to file.
    ///
    /// Returns `None` if the value of `header_len` is too large for this .npy version.
    fn format_header_len(self, header_len: usize) -> Option<Vec<u8>> {
        match self {
            Version::V1_0 => {
                let header_len: u16 = u16::try_from(header_len).ok()?;
                let mut out = vec![0; self.header_len_num_bytes()];
                LittleEndian::write_u16(&mut out, header_len);
                Some(out)
            }
            Version::V2_0 | Version::V3_0 => {
                let header_len: u32 = u32::try_from(header_len).ok()?;
                let mut out = vec![0; self.header_len_num_bytes()];
                LittleEndian::write_u32(&mut out, header_len);
                Some(out)
            }
        }
    }

    /// Computes the total header length, formatted `HEADER_LEN` value, and
    /// padding length for this .npy version.
    ///
    /// `unpadded_arr_format` is the Python literal describing the array
    /// format, formatted as an ASCII string without any padding.
    ///
    /// Returns `None` if the total header length overflows `usize` or if the
    /// value of `HEADER_LEN` is too large for this .npy version.
    fn compute_lengths(self, unpadded_arr_format: &[u8]) -> Option<HeaderLengthInfo> {
        /// Length of a '\n' char in bytes.
        const NEWLINE_LEN: usize = 1;

        let prefix_len: usize =
            MAGIC_STRING.len() + Version::VERSION_NUM_BYTES + self.header_len_num_bytes();
        let unpadded_total_len: usize = prefix_len
            .checked_add(unpadded_arr_format.len())?
            .checked_add(NEWLINE_LEN)?;
        let padding_len: usize = HEADER_DIVISOR - unpadded_total_len % HEADER_DIVISOR;
        let total_len: usize = unpadded_total_len.checked_add(padding_len)?;
        let header_len: usize = total_len - prefix_len;
        let formatted_header_len = self.format_header_len(header_len)?;
        Some(HeaderLengthInfo {
            total_len,
            formatted_header_len,
        })
    }
}

struct HeaderLengthInfo {
    /// Total header length (including magic string, version number, header
    /// length value, array format description, padding, and final newline).
    total_len: usize,
    /// Formatted `HEADER_LEN` value. (This is the number of bytes in the array
    /// format description, padding, and final newline.)
    formatted_header_len: Vec<u8>,
}

#[derive(Debug)]
pub enum FormatHeaderError {
    PyValue(PyValueFormatError),
    /// The total header length overflows `usize`, or `HEADER_LEN` exceeds the
    /// maximum encodable value.
    HeaderTooLong,
}

impl Error for FormatHeaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            FormatHeaderError::PyValue(err) => Some(err),
            FormatHeaderError::HeaderTooLong => None,
        }
    }
}

impl std::fmt::Display for FormatHeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            FormatHeaderError::PyValue(err) => write!(f, "error formatting Python value: {}", err),
            FormatHeaderError::HeaderTooLong => write!(f, "the header is too long"),
        }
    }
}

impl From<PyValueFormatError> for FormatHeaderError {
    fn from(err: PyValueFormatError) -> FormatHeaderError {
        FormatHeaderError::PyValue(err)
    }
}

#[derive(Debug)]
pub enum WriteHeaderError {
    Io(std::io::Error),
    Format(FormatHeaderError),
}

impl Error for WriteHeaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            WriteHeaderError::Io(err) => Some(err),
            WriteHeaderError::Format(err) => Some(err),
        }
    }
}

impl std::fmt::Display for WriteHeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            WriteHeaderError::Io(err) => write!(f, "I/O error: {}", err),
            WriteHeaderError::Format(err) => write!(f, "error formatting header: {}", err),
        }
    }
}

impl From<std::io::Error> for WriteHeaderError {
    fn from(err: std::io::Error) -> WriteHeaderError {
        WriteHeaderError::Io(err)
    }
}

impl From<FormatHeaderError> for WriteHeaderError {
    fn from(err: FormatHeaderError) -> WriteHeaderError {
        WriteHeaderError::Format(err)
    }
}

#[derive(Clone, Debug)]
pub struct Header {
    pub type_descriptor: PyValue,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

impl std::fmt::Display for Header {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "{}", self.to_py_value())
    }
}

impl Header {
    fn from_py_value(value: PyValue) -> Result<Self, ParseHeaderError> {
        if let PyValue::Dict(dict) = value {
            let mut type_descriptor: Option<PyValue> = None;
            let mut fortran_order: Option<bool> = None;
            let mut shape: Option<Vec<usize>> = None;
            for (key, value) in dict {
                match key {
                    PyValue::String(ref k) if k == "descr" => {
                        type_descriptor = Some(value);
                    }
                    PyValue::String(ref k) if k == "fortran_order" => {
                        if let PyValue::Boolean(b) = value {
                            fortran_order = Some(b);
                        } else {
                            return Err(ParseHeaderError::IllegalValue {
                                key: "fortran_order".to_owned(),
                                value,
                            });
                        }
                    }
                    PyValue::String(ref k) if k == "shape" => {
                        fn parse_shape(value: &PyValue) -> Option<Vec<usize>> {
                            value
                                .as_tuple()?
                                .iter()
                                .map(|elem| elem.as_integer()?.to_usize())
                                .collect()
                        }
                        if let Some(s) = parse_shape(&value) {
                            shape = Some(s);
                        } else {
                            return Err(ParseHeaderError::IllegalValue {
                                key: "shape".to_owned(),
                                value,
                            });
                        }
                    }
                    k => return Err(ParseHeaderError::UnknownKey(k)),
                }
            }
            match (type_descriptor, fortran_order, shape) {
                (Some(type_descriptor), Some(fortran_order), Some(shape)) => Ok(Header {
                    type_descriptor,
                    fortran_order,
                    shape,
                }),
                (None, _, _) => Err(ParseHeaderError::MissingKey("descr".to_owned())),
                (_, None, _) => Err(ParseHeaderError::MissingKey("fortran_order".to_owned())),
                (_, _, None) => Err(ParseHeaderError::MissingKey("shaper".to_owned())),
            }
        } else {
            Err(ParseHeaderError::MetaNotDict(value))
        }
    }

    pub fn from_reader<R: std::io::Read>(reader: &mut R) -> Result<Self, ReadHeaderError> {
        // Check for magic string.
        let mut buf = vec![0; MAGIC_STRING.len()];
        reader.read_exact(&mut buf)?;
        if buf != MAGIC_STRING {
            return Err(ParseHeaderError::MagicString.into());
        }

        // Get version number.
        let mut buf = [0; Version::VERSION_NUM_BYTES];
        reader.read_exact(&mut buf)?;
        let version = Version::from_bytes(&buf)?;

        // Get `HEADER_LEN`.
        let header_len = version.read_header_len(reader)?;

        // Parse the dictionary describing the array's format.
        let mut buf = vec![0; header_len];
        reader.read_exact(&mut buf)?;
        let without_newline = match buf.split_last() {
            Some((&b'\n', rest)) => rest,
            Some(_) | None => return Err(ParseHeaderError::MissingNewline.into()),
        };
        let header_str = match version {
            Version::V1_0 | Version::V2_0 => {
                if without_newline.is_ascii() {
                    // ASCII strings are always valid UTF-8.
                    unsafe { std::str::from_utf8_unchecked(without_newline) }
                } else {
                    return Err(ParseHeaderError::NonAscii.into());
                }
            }
            Version::V3_0 => {
                std::str::from_utf8(without_newline).map_err(ParseHeaderError::from)?
            }
        };
        let arr_format: PyValue = header_str.parse().map_err(ParseHeaderError::from)?;
        Ok(Header::from_py_value(arr_format)?)
    }

    fn to_py_value(&self) -> PyValue {
        PyValue::Dict(vec![
            (
                PyValue::String("descr".into()),
                self.type_descriptor.clone(),
            ),
            (
                PyValue::String("fortran_order".into()),
                PyValue::Boolean(self.fortran_order),
            ),
            (
                PyValue::String("shape".into()),
                PyValue::Tuple(
                    self.shape
                        .iter()
                        .map(|&elem| PyValue::Integer(elem.into()))
                        .collect(),
                ),
            ),
        ])
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, FormatHeaderError> {
        // Metadata describing array's format as ASCII string.
        let mut arr_format = Vec::new();
        self.to_py_value().write_ascii(&mut arr_format)?;

        // Determine appropriate version based on header length, and compute
        // length information.
        let (version, length_info) = [Version::V1_0, Version::V2_0]
            .iter()
            .find_map(|&version| Some((version, version.compute_lengths(&arr_format)?)))
            .ok_or(FormatHeaderError::HeaderTooLong)?;

        // Write the header.
        let mut out = Vec::with_capacity(length_info.total_len);
        out.extend_from_slice(MAGIC_STRING);
        out.push(version.major_version());
        out.push(version.minor_version());
        out.extend_from_slice(&length_info.formatted_header_len);
        out.extend_from_slice(&arr_format);
        out.resize(length_info.total_len - 1, b' ');
        out.push(b'\n');

        // Verify the length of the header.
        debug_assert_eq!(out.len(), length_info.total_len);
        debug_assert_eq!(out.len() % HEADER_DIVISOR, 0);

        Ok(out)
    }

    pub fn write<W: std::io::Write>(&self, mut writer: W) -> Result<(), WriteHeaderError> {
        let bytes = self.to_bytes()?;
        writer.write_all(&bytes)?;
        Ok(())
    }
}

/******************************************************************************/

impl From<ReadHeaderError> for crate::Error {
    fn from(error: ReadHeaderError) -> Self {
        match error {
            ReadHeaderError::Io(e) => crate::Error::Io(e),
            ReadHeaderError::Parse(e) => crate::Error::Serialization(e.to_string()),
        }
    }
}

impl From<WriteHeaderError> for crate::Error {
    fn from(error: WriteHeaderError) -> Self {
        match error {
            WriteHeaderError::Io(e) => crate::Error::Io(e),
            WriteHeaderError::Format(e) => crate::Error::Serialization(e.to_string()),
        }
    }
}
