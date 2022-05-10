use std::os::raw::c_char;
use std::ffi::CStr;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use super::status::{eqs_status_t, catch_unwind};
use super::tensor::eqs_tensormap_t;

/// Load a tensor map from the file at the given path.
///
/// The memory allocated by this function should be released using
/// `eqs_tensormap_free`.
///
/// @param path path to the file as a NULL-terminated UTF-8 string
///
/// @returns A pointer to the newly allocated tensor map, or a `NULL` pointer in
///          case of error. In case of error, you can use `eqs_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern fn eqs_tensormap_load(
    path: *const c_char,
) -> *mut eqs_tensormap_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        check_pointers!(path);

        let path = CStr::from_ptr(path).to_str().expect("use UTF-8 for path");
        let file = BufReader::new(File::open(path)?);
        let tensor = crate::io::load(file)?;
        let boxed = Box::new(eqs_tensormap_t(tensor));

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *(unwind_wrapper.0) = Box::into_raw(boxed);
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}


/// Save a tensor map to the file at the given path.
///
/// If the file already exists, it is overwritten.
///
/// @param path path to the file as a NULL-terminated UTF-8 string
/// @param tensor tensor map to save to the file
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_tensormap_save(
    path: *const c_char,
    tensor: *const eqs_tensormap_t,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(path, tensor);

        let path = CStr::from_ptr(path).to_str().expect("use UTF-8 for path");
        let file = BufWriter::new(File::create(path)?);
        crate::io::save(file, &*tensor)?;

        Ok(())
    })
}
