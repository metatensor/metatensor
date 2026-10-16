//! Possible ZIP compression methods.

use std::fmt;

#[allow(deprecated)]
/// Identifies the storage format used to compress a file within a ZIP archive.
///
/// Each file's compression method is stored alongside it, allowing the
/// contents to be read without context.
///
/// When creating ZIP files, you may choose the method to use with
/// [`crate::write::FileOptions::compression_method`]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub(crate) enum CompressionMethod {
    /// Store the file as is
    Stored,
    /// Compress the file using Deflate
    Deflated,
    /// Unsupported compression method
    #[deprecated(since = "0.5.7", note = "use the constants instead")]
    Unsupported(u16),
}
#[allow(deprecated, missing_docs)]
/// All compression methods defined for the ZIP format
impl CompressionMethod {
    pub const STORE: Self = CompressionMethod::Stored;
    pub const SHRINK: Self = CompressionMethod::Unsupported(1);
    pub const REDUCE_1: Self = CompressionMethod::Unsupported(2);
    pub const REDUCE_2: Self = CompressionMethod::Unsupported(3);
    pub const REDUCE_3: Self = CompressionMethod::Unsupported(4);
    pub const REDUCE_4: Self = CompressionMethod::Unsupported(5);
    pub const IMPLODE: Self = CompressionMethod::Unsupported(6);
    pub const DEFLATE: Self = CompressionMethod::Deflated;
    pub const DEFLATE64: Self = CompressionMethod::Unsupported(9);
    pub const PKWARE_IMPLODE: Self = CompressionMethod::Unsupported(10);
    pub const BZIP2: Self = CompressionMethod::Unsupported(12);
    pub const LZMA: Self = CompressionMethod::Unsupported(14);
    pub const IBM_ZOS_CMPSC: Self = CompressionMethod::Unsupported(16);
    pub const IBM_TERSE: Self = CompressionMethod::Unsupported(18);
    pub const ZSTD_DEPRECATED: Self = CompressionMethod::Unsupported(20);
    pub const ZSTD: Self = CompressionMethod::Unsupported(93);
    pub const MP3: Self = CompressionMethod::Unsupported(94);
    pub const XZ: Self = CompressionMethod::Unsupported(95);
    pub const JPEG: Self = CompressionMethod::Unsupported(96);
    pub const WAVPACK: Self = CompressionMethod::Unsupported(97);
    pub const PPMD: Self = CompressionMethod::Unsupported(98);
    pub const AES: Self = CompressionMethod::Unsupported(99);
}
impl CompressionMethod {
    /// Converts an u16 to its corresponding CompressionMethod
    #[deprecated(
        since = "0.5.7",
        note = "use a constant to construct a compression method"
    )]
    pub fn from_u16(val: u16) -> CompressionMethod {
        #[allow(deprecated)]
        match val {
            0 => CompressionMethod::Stored,
            8 => CompressionMethod::Deflated,
            v => CompressionMethod::Unsupported(v),
        }
    }

    /// Converts a CompressionMethod to a u16
    #[deprecated(
        since = "0.5.7",
        note = "to match on other compression methods, use a constant"
    )]
    pub fn to_u16(self) -> u16 {
        #[allow(deprecated)]
        match self {
            CompressionMethod::Stored => 0,
            CompressionMethod::Deflated => 8,
            CompressionMethod::Unsupported(v) => v,
        }
    }
}

impl fmt::Display for CompressionMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Just duplicate what the Debug format looks like, i.e, the enum key:
        write!(f, "{self:?}")
    }
}

/// The compression methods which have been implemented.
pub const SUPPORTED_COMPRESSION_METHODS: &[CompressionMethod] = &[
    CompressionMethod::Stored,
    CompressionMethod::Deflated,
];
