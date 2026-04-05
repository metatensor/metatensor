#![allow(clippy::needless_return)]

mod c_api;

pub use c_api::*;

/// Error type used in metatensor
#[derive(Debug, Clone)]
pub struct Error {
    /// Error code from the metatensor-core C API
    pub code: Option<mts_status_t>,
    /// Error message associated with the code
    pub message: String
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(code) = self.code {
            write!(f, " (code {})", code)?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

// Box<dyn Any + Send + 'static> is the error type in std::panic::catch_unwind
impl From<Box<dyn std::any::Any + Send + 'static>> for Error {
    fn from(error: Box<dyn std::any::Any + Send + 'static>) -> Error {
        let message = if let Some(message) = error.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = error.downcast_ref::<&str>() {
            (*message).to_owned()
        } else {
            panic!("panic message is not a string, something is very wrong")
        };

        Error {
            code: None, message,
        }
    }
}

impl mts_array_t {
    /// Create an `mts_array_t` with all members set to null pointers/None
    pub fn null() -> mts_array_t {
        mts_array_t {
            ptr: std::ptr::null_mut(),
            origin: None,
            device: None,
            dtype: None,
            as_dlpack: None,
            shape: None,
            reshape: None,
            swap_axes: None,
            create: None,
            copy: None,
            destroy: None,
            move_data: None,
        }
    }
}
