#![allow(clippy::doc_markdown)]
use std::ffi::CString;
use std::os::raw::c_char;

use once_cell::sync::Lazy;

#[macro_use]
mod status;
pub use self::status::{catch_unwind, mts_status_t};

#[cfg(test)]
pub use self::status::MTS_SUCCESS;

mod labels;
mod data;
mod blocks;
mod tensor;
mod io;

mod utils;

/// Disable printing of the message to stderr when some Rust code reach a panic.
///
/// All panics from Rust code are caught anyway and translated to an error
/// status code, and the message is stored and accessible through
/// `mts_last_error`. To print the error message and Rust backtrace anyway,
/// users can set the `RUST_BACKTRACE` environment variable to 1.
#[no_mangle]
pub extern fn mts_disable_panic_printing() {
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

static VERSION: Lazy<CString> = Lazy::new(|| {
    CString::new(env!("METATENSOR_FULL_VERSION")).expect("version contains NULL byte")
});


/// Get the version of the core metatensor library as a string.
///
/// This version should follow the `<major>.<minor>.<patch>[-<dev>]` format.
#[no_mangle]
pub extern fn mts_version() -> *const c_char {
    return VERSION.as_ptr();
}
