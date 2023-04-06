// This file was initially taken from https://github.com/jturner314/ndarray-npy,
// version 0.8.1. It is Copyright 2018–2021 Jim Turner and ndarray-npy
// developers, released under MIT and Apache Licenses.
use std::convert::TryFrom;
use std::error::Error;
use std::fmt::Write as FmtWrite;
use std::io::Write as IoWrite;

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};

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
    InvalidHeader(String),
}

impl Error for ParseHeaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ParseHeaderError::Utf8Parse(err) => Some(err),
            ParseHeaderError::MagicString |
            ParseHeaderError::Version { .. } |
            ParseHeaderError::HeaderLengthOverflow(_) |
            ParseHeaderError::NonAscii |
            ParseHeaderError::InvalidHeader(_) => None,
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
            ParseHeaderError::InvalidHeader(value) => write!(f, "invalid header in file: {}", value),
        }
    }
}

impl From<std::str::Utf8Error> for ParseHeaderError {
    fn from(err: std::str::Utf8Error) -> ParseHeaderError {
        ParseHeaderError::Utf8Parse(err)
    }
}

impl From<std::num::ParseIntError> for ParseHeaderError {
    fn from(e: std::num::ParseIntError) -> Self {
        ParseHeaderError::InvalidHeader(format!("failed to parse an integer: {}", e))
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
pub enum WriteHeaderError {
    Io(std::io::Error),
    Format(String),
}

impl Error for WriteHeaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            WriteHeaderError::Io(err) => Some(err),
            WriteHeaderError::Format(_) => None,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DataType {
    Scalar(String),
    Compound(Vec<(String, String)>),
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::Scalar(v) => write!(f, "'{}'", v),
            DataType::Compound(list) => {
                write!(f, "[")?;
                for (k, v) in list {
                    write!(f, "('{}', '{}'), ", k, v)?;
                }
                write!(f, "]")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub type_descriptor: DataType,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug)]
struct HeaderParser {
    data: Vec<char>,
    position: usize,
}

impl HeaderParser {
    fn done(&self) -> bool {
        return self.position >= self.data.len();
    }

    fn current(&self) -> char {
        return self.data[self.position];
    }

    fn advance(&mut self) -> char {
        let value = self.current();
        self.position += 1;
        return value;
    }

    fn expects(&mut self, c: char) -> Result<(), ParseHeaderError> {
        if self.current() == c {
            self.advance();
            return Ok(());
        } else {
            return Err(ParseHeaderError::InvalidHeader(format!(
                "expected '{}', got '{}'", c, self.current()
            )));
        }
    }

    fn skip_whitespaces(&mut self) {
        let mut c = self.current();
        while !self.done() && (c == ' ' || c == '\t' || c == '\x0C') {
            self.advance();
            c = self.current();
        }
    }

    fn parse_string(&mut self) -> Result<String, ParseHeaderError> {
        let mut value = String::new();
        if self.current() == '\'' {
            self.advance();
            while self.current() != '\'' {
                value.push(self.advance());
            }
            self.advance();

        } else if self.current() == '"' {
            self.advance();
            while self.current() != '"' {
                value.push(self.advance());
            }
            self.advance();
        } else {
            return Err(ParseHeaderError::InvalidHeader(format!(
                "expected a string, got '{}'", self.current()
            )));
        }

        return Ok(value);
    }

    fn parse_integer(&mut self) -> Result<usize, ParseHeaderError> {
        let mut value = String::new();
        loop {
            if self.current().is_ascii_digit() {
                value.push(self.advance());
            } else {
                break;
            }
        }

        if value.is_empty() {
            return Err(ParseHeaderError::InvalidHeader(format!(
                "expected an integer, got '{}'", self.current()
            )));
        }

        return Ok(value.parse()?);
    }

    fn parse_data_type(&mut self) -> Result<DataType, ParseHeaderError> {
        if self.current() == '\'' || self.current() == '"' {
            let value = self.parse_string()?;
            return Ok(DataType::Scalar(value));
        } else if self.current() == '[' {
            self.advance();

            let mut data_type = Vec::new();
            loop {
                self.skip_whitespaces();
                self.expects('(')?;
                self.skip_whitespaces();

                let name = self.parse_string()?;

                self.skip_whitespaces();
                self.expects(',')?;
                self.skip_whitespaces();

                let value = self.parse_string()?;

                self.skip_whitespaces();
                self.expects(')')?;
                self.skip_whitespaces();

                data_type.push((name, value));

                if self.current() == ',' {
                    self.advance();
                    self.skip_whitespaces();
                } else {
                    self.expects(']')?;
                    break;
                }

                if self.current() == ']' {
                    self.advance();
                    break;
                }
            }

            return Ok(DataType::Compound(data_type));
        } else {
            return Err(ParseHeaderError::InvalidHeader(format!(
                "expected a string or a list, got '{}'", self.current()
            )));
        }
    }

    fn parse_bool(&mut self) -> Result<bool, ParseHeaderError> {
        if self.current() == 'T' {
            self.advance();
            self.expects('r')?;
            self.expects('u')?;
            self.expects('e')?;
            return Ok(true);
        } else if self.current() == 'F' {
            self.advance();
            self.expects('a')?;
            self.expects('l')?;
            self.expects('s')?;
            self.expects('e')?;
            return Ok(false);
        } else {
            return Err(ParseHeaderError::InvalidHeader(format!(
                "expected a bool, got '{}'", self.current()
            )));
        }
    }

    fn parse_shape(&mut self) -> Result<Vec<usize>, ParseHeaderError> {
        let mut shape = Vec::new();
        self.expects('(')?;
        loop {
            self.skip_whitespaces();
            shape.push(self.parse_integer()?);
            self.skip_whitespaces();


            if self.current() == ',' {
                self.advance();
                self.skip_whitespaces();
            } else {
                self.expects(')')?;
                break;
            }

            if self.current() == ')' {
                self.advance();
                break;
            }
        }

        return Ok(shape);
    }

    fn parse(&mut self) -> Result<Header, ParseHeaderError> {
        let mut type_descriptor: Option<DataType> = None;
        let mut fortran_order: Option<bool> = None;
        let mut shape: Option<Vec<usize>> = None;

        self.skip_whitespaces();
        self.expects('{')?;
        self.skip_whitespaces();

        loop {
            let key = self.parse_string()?;
            self.skip_whitespaces();
            self.expects(':')?;
            self.skip_whitespaces();

            if key == "descr" {
                type_descriptor = Some(self.parse_data_type()?);
            } else if key == "fortran_order" {
                fortran_order = Some(self.parse_bool()?);
            } else if key == "shape" {
                shape = Some(self.parse_shape()?);
            } else {
                return Err(ParseHeaderError::InvalidHeader(format!(
                    "unknown key: '{}'", key
                )));
            }

            self.skip_whitespaces();
            if self.current() == ',' {
                self.advance();
                self.skip_whitespaces();
            } else {
                self.expects('}')?;
                break;
            }

            if self.current() == '}' {
                self.advance();
                break;
            }
        }

        match (type_descriptor, fortran_order, shape) {
            (Some(type_descriptor), Some(fortran_order), Some(shape)) => Ok(Header {
                type_descriptor,
                fortran_order,
                shape,
            }),
            (None, _, _) => Err(ParseHeaderError::InvalidHeader("missing 'descr' key".into())),
            (_, None, _) => Err(ParseHeaderError::InvalidHeader("missing 'fortran_order' key".into())),
            (_, _, None) => Err(ParseHeaderError::InvalidHeader("missing 'shape' key".into())),
        }
    }
}

impl Header {
    fn from_str(value: &str) -> Result<Self, ParseHeaderError> {
        let mut parser = HeaderParser { data: value.chars().collect(), position: 0 };
        return parser.parse();
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
            Some(_) | None => return Err(ParseHeaderError::InvalidHeader("missing new line".into()))?,
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
        // let arr_format: PyValue = header_str.parse().map_err(ParseHeaderError::from)?;
        Ok(Header::from_str(header_str)?)
    }

    fn to_dict_literal(&self) -> String {
        let mut result = String::new();
        write!(&mut result, "{{ 'descr': {}, ", self.type_descriptor).expect("failed to write");

        let order = if self.fortran_order {
            "True"
        } else {
            "False"
        };
        write!(&mut result, "'fortran_order': {}, ", order).expect("failed to write");

        write!(&mut result, "'shape': (").expect("failed to write");
        for s in &self.shape {
            write!(&mut result, "{}, ", s).expect("failed to write");
        }
        write!(&mut result, ") }}").expect("failed to write");
        return result;
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, WriteHeaderError> {
        // Metadata describing array's format as ASCII string.
        let mut arr_format = Vec::new();

        write!(&mut arr_format, "{}", self.to_dict_literal())?;

        // Determine appropriate version based on header length, and compute
        // length information.
        let (version, length_info) = [Version::V1_0, Version::V2_0]
            .iter()
            .find_map(|&version| Some((version, version.compute_lengths(&arr_format)?)))
            .ok_or_else(|| WriteHeaderError::Format("header too long".into()))?;

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
            WriteHeaderError::Format(e) => crate::Error::Serialization(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_header_parsing() {
        let header = "  \t{'descr': [('a', '<i4'),('b', '<i4'),], 'shape': (27, 3), 'fortran_order':False, }";

        let header = Header::from_str(header).unwrap();

        assert_eq!(header.type_descriptor, DataType::Compound(
            vec![("a".into(), "<i4".into()), ("b".into(), "<i4".into())]
        ));
        assert!(!header.fortran_order);
        assert_eq!(header.shape, [27, 3]);

        assert_eq!(header.to_dict_literal(), "{ 'descr': [('a', '<i4'), ('b', '<i4'), ], 'fortran_order': False, 'shape': (27, 3, ) }");

        let header = Header {
            type_descriptor: DataType::Scalar("<f8".into()),
            fortran_order: true,
            shape: vec![3],
        };
        assert_eq!(header.to_dict_literal(), "{ 'descr': '<f8', 'fortran_order': True, 'shape': (3, ) }");

        // check round trip
        let parsed = Header::from_str(&header.to_dict_literal()).unwrap();
        assert_eq!(parsed, header);
    }
}
