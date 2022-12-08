use std::panic::UnwindSafe;
use std::cell::RefCell;
use std::os::raw::c_char;
use std::ffi::CString;

use crate::Error;

// Save the last error message in thread local storage.
//
// This is marginally better than a standard global static value because it
// allow multiple threads to each have separate errors conditions.
thread_local! {
    pub static LAST_ERROR_MESSAGE: RefCell<CString> = RefCell::new(CString::new("").expect("invalid C string"));
}

/// Status type returned by all functions in the C API.
///
/// The value 0 (`EQS_SUCCESS`) is used to indicate successful operations,
/// positive values are used by this library to indicate errors, while negative
/// values are reserved for users of this library to indicate their own errors
/// in callbacks.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
#[must_use]
pub struct eqs_status_t(pub(crate) i32);

impl eqs_status_t {
    pub fn is_success(self) -> bool {
        self.0 == EQS_SUCCESS
    }

    pub fn as_i32(self) -> i32 {
        self.0
    }
}

/// Status code used when a function succeeded
pub const EQS_SUCCESS: i32 = 0;
/// Status code used when a function got an invalid parameter
pub const EQS_INVALID_PARAMETER_ERROR: i32 = 1;
/// Status code indicating I/O error when loading/writing `eqs_tensormap_t` to a file
pub const EQS_IO_ERROR: i32 = 2;
/// Status code indicating errors in the serialization format when
/// loading/writing `eqs_tensormap_t` to a file
pub const EQS_SERIALIZATION_ERROR: i32 = 3;

/// Status code used when a memory buffer is too small to fit the requested data
pub const EQS_BUFFER_SIZE_ERROR: i32 = 254;
/// Status code used when there was an internal error, i.e. there is a bug
/// inside equistore itself
pub const EQS_INTERNAL_ERROR: i32 = 255;


impl From<Error> for eqs_status_t {
    #[allow(clippy::match_same_arms)]
    fn from(error: Error) -> eqs_status_t {
        LAST_ERROR_MESSAGE.with(|message| {
            *message.borrow_mut() = CString::new(format!("{}", error)).expect("error message contains a null byte");
        });
        match error {
            Error::InvalidParameter(_) => eqs_status_t(EQS_INVALID_PARAMETER_ERROR),
            Error::Io(_) => eqs_status_t(EQS_IO_ERROR),
            Error::Serialization(_) => eqs_status_t(EQS_SERIALIZATION_ERROR),
            Error::BufferSize(_) => eqs_status_t(EQS_BUFFER_SIZE_ERROR),
            Error::External {status, .. } => status,
            Error::Internal(_) => eqs_status_t(EQS_INTERNAL_ERROR),
        }
    }
}

/// An alternative to `std::panic::catch_unwind` that automatically transform
/// the error into `eqs_status_t`.
pub fn catch_unwind<F>(function: F) -> eqs_status_t where F: FnOnce() -> Result<(), Error> + UnwindSafe {
    match std::panic::catch_unwind(function) {
        Ok(Ok(_)) => eqs_status_t(EQS_SUCCESS),
        Ok(Err(error)) => error.into(),
        Err(error) => Error::from(error).into()
    }
}

/// Check that pointers (used as C API function parameters) are not null.
#[macro_export]
#[doc(hidden)]
macro_rules! check_pointers {
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
        $(check_pointers!($pointer);)*
    }
}

/// Get the last error message that was created on the current thread.
///
/// @returns the last error message, as a NULL-terminated string
#[no_mangle]
pub unsafe extern fn eqs_last_error() -> *const c_char {
    let mut result = std::ptr::null();
    let wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        let wrapper = wrapper;
        LAST_ERROR_MESSAGE.with(|message| {
            *wrapper.0 = message.borrow().as_ptr();
        });
        Ok(())
    });

    if status.0 != EQS_SUCCESS {
        eprintln!("ERROR: unable to get last error message!");
        return std::ptr::null();
    }

    return result;
}
