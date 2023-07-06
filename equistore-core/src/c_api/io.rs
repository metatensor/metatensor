use std::os::raw::c_char;
use std::ffi::CStr;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use crate::Error;
use crate::data::eqs_array_t;

use super::status::{eqs_status_t, catch_unwind};
use super::tensor::eqs_tensormap_t;

/// Function pointer to create a new `eqs_array_t` when de-serializing tensor
/// maps.
///
/// This function gets the `shape` of the array (the `shape` contains
/// `shape_count` elements) and should fill `array` with a new valid
/// `eqs_array_t` or return non-zero `eqs_status_t`.
///
/// The newly created array should contains 64-bit floating points (`double`)
/// data, and live on CPU, since equistore will use `eqs_array_t.data` to get
/// the data pointer and write to it.
#[allow(non_camel_case_types)]
type eqs_create_array_callback_t = unsafe extern fn(
    shape: *const usize,
    shape_count: usize,
    array: *mut eqs_array_t,
) -> eqs_status_t;

/// Load a tensor map from the file at the given path.
///
/// Arrays for the values and gradient data will be created with the given
/// `create_array` callback, and filled by this function with the corresponding
/// data.
///
/// The memory allocated by this function should be released using
/// `eqs_tensormap_free`.
///
/// `TensorMap` are serialized using numpy's `.npz` format, i.e. a ZIP file
/// without compression (storage method is STORED), where each file is stored as
/// a `.npy` array. Both the ZIP and NPY format are well documented:
///
/// - ZIP: <https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT>
/// - NPY: <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>
///
/// We add other restriction on top of these formats when saving/loading data.
/// First, `Labels` instances are saved as structured array, see the `labels`
/// module for more information. Only 32-bit integers are supported for Labels,
/// and only 64-bit floats are supported for data (values and gradients).
///
/// Second, the path of the files in the archive also carry meaning. The keys of
/// the `TensorMap` are stored in `/keys.npy`, and then different blocks are
/// stored as
///
/// ```bash
/// /  blocks / <block_id>  / values / samples.npy
///                         / values / components  / 0.npy
///                                                / <...>.npy
///                                                / <n_components>.npy
///                         / values / properties.npy
///                         / values / data.npy
///
///                         # optional sections for gradients, one by parameter
///                         /   gradients / <parameter> / samples.npy
///                                                     /   components  / 0.npy
///                                                                     / <...>.npy
///                                                                     / <n_components>.npy
///                                                     /   data.npy
/// ```
///
/// @param path path to the file as a NULL-terminated UTF-8 string
/// @param create_array callback function that will be used to create data
///                     arrays inside each block
///
/// @returns A pointer to the newly allocated tensor map, or a `NULL` pointer in
///          case of error. In case of error, you can use `eqs_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern fn eqs_tensormap_load(
    path: *const c_char,
    create_array: eqs_create_array_callback_t,
) -> *mut eqs_tensormap_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        check_pointers!(path);

        let create_array = wrap_create_array(&create_array);

        let path = CStr::from_ptr(path).to_str().expect("use UTF-8 for path");
        let file = BufReader::new(File::open(path)?);
        let tensor = crate::io::load(file, create_array)?;

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *(unwind_wrapper.0) = eqs_tensormap_t::into_boxed_raw(tensor);
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}

/// Load a tensor map from the given in-memory buffer.
///
/// Arrays for the values and gradient data will be created with the given
/// `create_array` callback, and filled by this function with the corresponding
/// data.
///
/// The memory allocated by this function should be released using
/// `eqs_tensormap_free`.
///
/// @param buffer buffer containing a previously serialized `eqs_tensormap_t`
/// @param buffer_count number of elements in the buffer
/// @param create_array callback function that will be used to create data
///                     arrays inside each block
///
/// @returns A pointer to the newly allocated tensor map, or a `NULL` pointer in
///          case of error. In case of error, you can use `eqs_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern fn eqs_tensormap_load_buffer(
    buffer: *const u8,
    buffer_count: usize,
    create_array: eqs_create_array_callback_t,
) -> *mut eqs_tensormap_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        check_pointers!(buffer);
        assert!(buffer_count > 0);

        let create_array = wrap_create_array(&create_array);

        let buffer = std::slice::from_raw_parts(buffer.cast::<u8>(), buffer_count);
        let cursor = std::io::Cursor::new(buffer);

        let tensor = crate::io::load(cursor, create_array)?;

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *(unwind_wrapper.0) = eqs_tensormap_t::into_boxed_raw(tensor);
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}

fn wrap_create_array(create_array: &eqs_create_array_callback_t) -> impl Fn(Vec<usize>) -> Result<eqs_array_t, Error> + '_ {
    |shape: Vec<usize>| {
        let mut array = eqs_array_t::null();
        let status = unsafe {
            create_array(
                shape.as_ptr(),
                shape.len(),
                &mut array
            )
        };

        if status.is_success() {
            return Ok(array);
        } else {
            return Err(Error::External {
                status: status,
                context: "failed to create a new array in eqs_tensormap_load".into()
            });
        }
    }
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
