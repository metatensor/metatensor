#![allow(clippy::doc_markdown)]

#[macro_use]
mod status;
pub use self::status::{catch_unwind, eqs_status_t};
pub use self::status::{EQS_SUCCESS, EQS_INVALID_PARAMETER_ERROR};
pub use self::status::{EQS_BUFFER_SIZE_ERROR, EQS_INTERNAL_ERROR};

pub mod labels;
pub use self::labels::eqs_labels_t;

pub mod data;

pub mod blocks;
pub use self::blocks::eqs_block_t;

pub mod tensor;
pub use self::tensor::eqs_tensormap_t;

pub mod io;

mod utils;

/// Get the content of the C API header for aml_storage functions & structs
pub fn header_content() -> &'static str {
    include_str!("../../include/equistore.h")
}
