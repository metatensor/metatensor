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


/// Disable printing of the message to stderr when some Rust code reach a panic.
///
/// All panics from Rust code are caught anyway and translated to an error
/// status code, and the message is stored and accessible through
/// `eqs_last_error`. To print the error message and Rust backtrace anyway,
/// users can set the `RUST_BACKTRACE` environment variable to 1.
#[no_mangle]
pub extern fn eqs_disable_panic_printing() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        match std::env::var("RUST_BACKTRACE") {
            Ok(v) if v == "0" => {}
            Ok(_) => {
                // is RUST_BACKTRACE is set to a non 0 value, call the default
                // panic handler
                previous(info);
            }
            _ => {}
        }
    }));
}
