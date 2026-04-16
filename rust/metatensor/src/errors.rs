use std::ffi::CStr;

use crate::c_api::{mts_status_t, MTS_SUCCESS, MTS_CALLBACK_ERROR};

pub use metatensor_sys::Error;

fn get_last_error(status: Option<mts_status_t>) -> Error {
    let mut message = std::ptr::null();
    let mut origin = std::ptr::null();
    let mut user_data = std::ptr::null_mut();
    let last_error_status = unsafe {
        crate::c_api::mts_last_error(
             &mut message, &mut origin, &mut user_data
        )
    };

    if last_error_status != MTS_SUCCESS {
        return Error {
            code: status,
            message: "INTERNAL ERROR: failed to get the last error".into(),
        };
    }

    let message = if message.is_null() {
        "<no message provided>"
    } else {
        unsafe { CStr::from_ptr(message).to_str().unwrap_or("<invalid UTF-8 in error message>") }
    };

    let origin = if origin.is_null() {
        "<no origin provided>"
    } else {
        unsafe { CStr::from_ptr(origin).to_str().unwrap_or("<invalid UTF-8 in error origin>") }
    };

    if !user_data.is_null() && origin == "Rust Error" {
        let rust_error = unsafe {
            user_data.cast::<Error>().as_ref().expect("should not be null")
        };
        return rust_error.clone();
    }

    return Error {
        code: status,
        message: message.to_owned(),
    };
}

unsafe extern "C" fn error_deleter(data: *mut std::ffi::c_void) {
    let _ = unsafe { Box::from_raw(data.cast::<Error>()) };
}

fn store_last_error(error: Error) -> mts_status_t {
    let c_message = std::ffi::CString::new(error.message.clone()).expect("found NULL byte in error message");
    let c_origin = std::ffi::CString::new("Rust Error").expect("found NULL byte in error origin");
    let status = unsafe {
        crate::c_api::mts_set_last_error(
            c_message.as_ptr(),
            c_origin.as_ptr(),
            Box::into_raw(Box::new(error)).cast(),
            Some(error_deleter),
        )
    };

    check_status(status).expect("failed to set last error");

    return MTS_CALLBACK_ERROR;
}

/// Check an `mts_status_t`, returning an error if is it not `MTS_SUCCESS`
pub fn check_status(status: mts_status_t) -> Result<(), Error> {
    if status == MTS_SUCCESS {
        return Ok(())
    } else {
        return Err(get_last_error(Some(status)));
    }
}

/// Check a pointer allocated by metatensor-core, returning an error if is null
pub fn check_ptr<T>(ptr: *const T) -> Result<(), Error> {
    if ptr.is_null() {
        return Err(get_last_error(None));
    }

    return Ok(())
}


/// An alternative to `std::panic::catch_unwind` that automatically transform
/// the error into `mts_status_t`.
pub(crate) fn catch_unwind<F>(function: F) -> mts_status_t where F: FnOnce() -> Result<(), Error> + std::panic::UnwindSafe {
    match std::panic::catch_unwind(function) {
        Ok(Ok(())) => MTS_SUCCESS,
        Ok(Err(e)) => {
            return store_last_error(e);
        },
        Err(e) => {
            return store_last_error(e.into());
        }
    }
}
