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
/// The value 0 (`AML_SUCCESS`) is used to indicate successful operations.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
#[must_use]
pub struct aml_status_t(i32);

impl aml_status_t {
    pub fn is_success(self) -> bool {
        self.0 == AML_SUCCESS
    }

    pub fn as_i32(self) -> i32 {
        self.0
    }
}

/// Status code used when a function succeeded
pub const AML_SUCCESS: i32 = 0;
/// Status code used when a function got an invalid parameter
pub const AML_INVALID_PARAMETER_ERROR: i32 = 1;
/// Status code used when a memory buffer is too small to fit the requested data
pub const AML_BUFFER_SIZE_ERROR: i32 = 254;
/// Status code used when there was an internal error, i.e. there is a bug
/// inside AML itself
pub const AML_INTERNAL_ERROR: i32 = 255;


impl From<Error> for aml_status_t {
    #[allow(clippy::match_same_arms)]
    fn from(error: Error) -> aml_status_t {
        LAST_ERROR_MESSAGE.with(|message| {
            *message.borrow_mut() = CString::new(format!("{}", error)).expect("error message contains a null byte");
        });
        match error {
            Error::InvalidParameter(_) => aml_status_t(AML_INVALID_PARAMETER_ERROR),
            Error::BufferSize(_) => aml_status_t(AML_BUFFER_SIZE_ERROR),
            Error::Internal(_) => aml_status_t(AML_INTERNAL_ERROR),
        }
    }
}

/// An alternative to `std::panic::catch_unwind` that automatically transform
/// the error into `aml_status_t`.
pub fn catch_unwind<F>(function: F) -> aml_status_t where F: FnOnce() -> Result<(), Error> + UnwindSafe {
    match std::panic::catch_unwind(function) {
        Ok(Ok(_)) => aml_status_t(AML_SUCCESS),
        Ok(Err(error)) => error.into(),
        Err(error) => Error::from(error).into()
    }
}

/// Check that pointers (used as C API function parameters) are not null.
#[macro_export]
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
pub unsafe extern fn aml_last_error() -> *const c_char {
    let mut result = std::ptr::null();
    let wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        let wrapper = wrapper;
        LAST_ERROR_MESSAGE.with(|message| {
            *wrapper.0 = message.borrow().as_ptr();
        });
        Ok(())
    });

    if status.0 != AML_SUCCESS {
        eprintln!("ERROR: unable to get last error message!");
        return std::ptr::null();
    }

    return result;
}
