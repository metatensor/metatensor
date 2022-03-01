
#[macro_use]
mod status;
pub use self::status::{catch_unwind, aml_status_t};
pub use self::status::{AML_SUCCESS, AML_INVALID_PARAMETER_ERROR};
pub use self::status::{AML_BUFFER_SIZE_ERROR, AML_INTERNAL_ERROR};

pub mod indexes;
pub mod data;
pub mod blocks;
pub mod descriptor;

mod utils;
