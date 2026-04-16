use std::panic::UnwindSafe;
use std::cell::RefCell;
use std::os::raw::{c_char, c_void};
use std::ffi::{CString, CStr};

use crate::Error;

#[derive(Debug)]
struct LastError {
    message: CString,
    origin: CString,
    custom_data: *mut c_void,
    custom_data_deleter: Option<unsafe extern "C" fn(*mut c_void)>,
}

impl std::ops::Drop for LastError {
    fn drop(&mut self) {
        if let Some(deleter) = self.custom_data_deleter {
            unsafe { deleter(self.custom_data) };
        }
    }
}

// Save the last error message in thread local storage.
//
// This is marginally better than a standard global static value because it
// allow multiple threads to each have separate errors conditions.
thread_local! {
    pub static LAST_ERROR: RefCell<LastError> = RefCell::new(LastError {
        message: CString::new("").expect("invalid C string"),
        origin: CString::new("").expect("invalid C string"),
        custom_data: std::ptr::null_mut(),
        custom_data_deleter: None,
    });
}

/// Add some context to the last error message, by prefixing it with the provided
pub(crate) fn add_error_context(context: &str) {
    LAST_ERROR.with(|last_error| {
        let mut last_error = last_error.borrow_mut();
        let current_message = last_error.message.to_str().unwrap_or("<invalid UTF-8 in error message>");
        last_error.message = CString::new(
            format!("{}: {}", context, current_message)
        ).expect("error message contains a null byte");
    });
}

/// Status type returned by all functions in the C API.
///
/// The value 0 (`MTS_SUCCESS`) is used to indicate successful operations.
#[repr(i32)]
#[must_use]
#[non_exhaustive]
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum mts_status_t {
    /// Status code used when a function succeeded
    MTS_SUCCESS = 0,
    /// Status code used when a function got an invalid parameter
    MTS_INVALID_PARAMETER_ERROR = 1,
    /// Status code indicating I/O error when loading/writing `mts_tensormap_t`
    /// to a file
    MTS_IO_ERROR = 2,
    /// Status code indicating errors in the serialization format when
    /// loading/writing `mts_tensormap_t` to a file
    MTS_SERIALIZATION_ERROR = 3,
    /// Status code used when a memory buffer is too small to fit the requested
    /// data
    MTS_BUFFER_SIZE_ERROR = 4,
    /// Status code indicating errors that comes from callbacks provided by the
    /// user of metatensor. The error message and arbitrary custom data can be
    /// stored using `mts_set_last_error` inside the callback, and retrieved
    /// later with `mts_last_error`.
    MTS_CALLBACK_ERROR = 254,
    /// Status code used when there was an internal error, i.e. there is a bug
    /// inside metatensor itself
    MTS_INTERNAL_ERROR = 255,
}

impl mts_status_t {
    pub fn is_success(self) -> bool {
        self == mts_status_t::MTS_SUCCESS
    }
}

impl std::fmt::Display for mts_status_t {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // this is an if instead of a match to allow the catch-all case
        // at the end, since we can't enforce that callbacks don't return
        // arbitrary status codes.
        let code = *self as i32;
        if code == mts_status_t::MTS_SUCCESS as i32 {
            write!(f, "MTS_SUCCESS")
        } else if code == mts_status_t::MTS_INVALID_PARAMETER_ERROR as i32 {
            write!(f, "MTS_INVALID_PARAMETER_ERROR")
        } else if code == mts_status_t::MTS_IO_ERROR as i32 {
            write!(f, "MTS_IO_ERROR")
        } else if code == mts_status_t::MTS_SERIALIZATION_ERROR as i32 {
            write!(f, "MTS_SERIALIZATION_ERROR")
        } else if code == mts_status_t::MTS_BUFFER_SIZE_ERROR as i32 {
            write!(f, "MTS_BUFFER_SIZE_ERROR")
        } else if code == mts_status_t::MTS_CALLBACK_ERROR as i32 {
            write!(f, "MTS_CALLBACK_ERROR")
        } else if code == mts_status_t::MTS_INTERNAL_ERROR as i32 {
            write!(f, "MTS_INTERNAL_ERROR")
        } else {
            write!(f, "unknown status code {}", code)
        }
    }
}

impl From<Error> for mts_status_t {
    #[allow(clippy::match_same_arms)]
    fn from(error: Error) -> mts_status_t {
        if let Error::CallbackError = error {
            // if the error is already a callback error, keep the original
            // status code and LAST_ERROR
            return mts_status_t::MTS_CALLBACK_ERROR;
        }

        LAST_ERROR.set(LastError {
            message: CString::new(format!("{}", error)).expect("error message contains a null byte"),
            origin: CString::new("metatensor-core").expect("invalid C string"),
            custom_data: std::ptr::null_mut(),
            custom_data_deleter: None,
        });

        match error {
            Error::InvalidParameter(_) => mts_status_t::MTS_INVALID_PARAMETER_ERROR,
            Error::Io(_) => mts_status_t::MTS_IO_ERROR,
            Error::Serialization(_) => mts_status_t::MTS_SERIALIZATION_ERROR,
            Error::BufferSize(_) => mts_status_t::MTS_BUFFER_SIZE_ERROR,
            Error::CallbackError => unreachable!(),
            Error::Internal(_) => mts_status_t::MTS_INTERNAL_ERROR,
        }
    }
}

/// An alternative to `std::panic::catch_unwind` that automatically transform
/// the error into `mts_status_t`.
pub fn catch_unwind<F>(function: F) -> mts_status_t where F: FnOnce() -> Result<(), Error> + UnwindSafe {
    match std::panic::catch_unwind(function) {
        Ok(Ok(())) => mts_status_t::MTS_SUCCESS,
        Ok(Err(error)) => error.into(),
        Err(error) => Error::from(error).into()
    }
}

/// Check that pointers (used as C API function parameters) are not null.
#[macro_export]
#[doc(hidden)]
macro_rules! check_pointers_non_null {
    ($pointer: ident) => {
        if $pointer.is_null() {
            return Err($crate::Error::InvalidParameter(
                format!(
                    "got invalid NULL pointer for {} at {}:{}",
                    stringify!($pointer), file!(), line!()
                )
            ));
        }
    };
    ($($pointer: ident),* $(,)?) => {
        $(check_pointers_non_null!($pointer);)*
    }
}

/// Get the last error message that was created on the current thread.
///
/// @param message if not NULL, this will be set to the last error message, as a
///        NULL-terminated string
/// @param origin if not NULL, this will be set to the origin of the last error,
///        as a NULL-terminated string
/// @param data if not NULL, this will be set to the custom data of the last error
///
/// @returns The status code of this operation.
#[no_mangle]
pub unsafe extern "C" fn mts_last_error(
    message: *mut *const c_char,
    origin: *mut *const c_char,
    data: *mut *mut c_void,
) -> mts_status_t {
    let status = std::panic::catch_unwind(|| {
        LAST_ERROR.with(|last_error| {
            if !message.is_null() {
                *message = last_error.borrow().message.as_ptr();
            }
            if !origin.is_null() {
                *origin = last_error.borrow().origin.as_ptr();
            }
            if !data.is_null() {
                *data = last_error.borrow().custom_data;
            }
        });
    });

    match status {
        Ok(()) => mts_status_t::MTS_SUCCESS,
        Err(error) => {
            // something went very wrong, try to print a message to stderr
            let last_error_debug = LAST_ERROR.with(|last_error| {
                format!("{:?}", last_error.borrow())
            });
            if error.is::<String>() {
                eprintln!("panic in mts_last_error: {:?}, last_error: {:?}", error.downcast_ref::<String>(), last_error_debug);
            } else if error.is::<&str>() {
                eprintln!("panic in mts_last_error: {:?}, last_error: {:?}", error.downcast_ref::<&str>(), last_error_debug);
            } else {
                eprintln!("panic in mts_last_error: unknown panic error type. last_error: {:?}", last_error_debug);
            }
            mts_status_t::MTS_INTERNAL_ERROR
        }
    }
}

/// Set the last error message for the current thread.
///
/// This is useful when the error is created in a callback provided by the user
/// of metatensor.
///
/// @param message the error message to set, as a NULL-terminated string
#[no_mangle]
pub unsafe extern "C" fn mts_set_last_error(
    message: *const c_char,
    origin: *const c_char,
    data: *mut c_void,
    data_deleter: Option<unsafe extern "C" fn(*mut c_void)>,
) -> mts_status_t {
    catch_unwind(move || {
        let message = if message.is_null() {
            CString::new("<no message provided>").expect("invalid C string")
        } else {
            CString::from(CStr::from_ptr(message))
        };

        let origin = if origin.is_null() {
            CString::new("<no origin provided>").expect("invalid C string")
        } else {
            CString::from(CStr::from_ptr(origin))
        };

        LAST_ERROR.set(LastError {
            message: message,
            origin: origin,
            custom_data: data,
            custom_data_deleter: data_deleter,
        });
        Ok(())
    })
}
