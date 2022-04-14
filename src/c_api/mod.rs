#![allow(clippy::doc_markdown)]

#[macro_use]
mod status;
pub use self::status::{catch_unwind, eqs_status_t};
pub use self::status::{EQS_SUCCESS, EQS_INVALID_PARAMETER_ERROR};
pub use self::status::{EQS_BUFFER_SIZE_ERROR, EQS_INTERNAL_ERROR};

pub mod labels;
pub mod data;
pub mod blocks;
pub mod tensor;

mod utils;
