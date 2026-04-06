
use std::os::raw::{c_char, c_void};
use std::sync::Arc;
use std::ffi::CStr;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use crate::Error;

use super::{ExternalBuffer, mts_realloc_buffer_t};

use super::super::status::{mts_status_t, catch_unwind};
use super::super::labels::mts_labels_t;

/// Load labels from the file at the given path.
///
/// This function allocates memory which must be released with
/// `mts_labels_free` when you don't need it anymore.
///
/// @param path path to the file as a NULL-terminated UTF-8 string
///
/// @returns A pointer to the newly allocated labels, or a `NULL` pointer in
///          case of error. In case of error, you can use `mts_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern "C" fn mts_labels_load(
    path: *const c_char,
) -> *const mts_labels_t {
    let mut result = std::ptr::null();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);

    let status = catch_unwind(move || {
        check_pointers_non_null!(path);

        let path = CStr::from_ptr(path).to_str().expect("use UTF-8 for path");
        let file = BufReader::new(File::open(path)?);

        let rust_labels = crate::io::load_labels(file)
            .map_err(|err| match err {
                Error::Serialization(message) => {
                    if crate::io::looks_like_tensormap_data(crate::io::PathOrBuffer::Path(path)) {
                        Error::Serialization(format!(
                            "unable to load Labels from '{}', use `load` to load TensorMap: {}", path, message
                        ))
                    } else if crate::io::looks_like_block_data(crate::io::PathOrBuffer::Path(path)) {
                        Error::Serialization(format!(
                            "unable to load Labels from '{}', use `load_block` to load TensorBlock: {}", path, message
                        ))
                    } else {
                        Error::Serialization(format!(
                            "unable to load Labels from '{}': {}", path, message
                        ))
                    }
                }
                err => return err,
            })?;

        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = mts_labels_t::into_raw(Arc::new(rust_labels));

        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}

/// Load labels from the given in-memory buffer.
///
/// This function allocates memory which must be released with
/// `mts_labels_free` when you don't need it anymore.
///
/// @param buffer buffer containing a previously serialized `mts_labels_t`
/// @param buffer_count number of elements in the buffer
///
/// @returns A pointer to the newly allocated labels, or a `NULL` pointer in
///          case of error. In case of error, you can use `mts_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern "C" fn mts_labels_load_buffer(
    buffer: *const u8,
    buffer_count: usize,
) -> *const mts_labels_t {
    let mut result = std::ptr::null();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);

    let status = catch_unwind(move || {
        check_pointers_non_null!(buffer);

        let slice = std::slice::from_raw_parts(buffer.cast::<u8>(), buffer_count);
        let cursor = std::io::Cursor::new(slice);

        let rust_labels = crate::io::load_labels(cursor)
            .map_err(|err| match err {
                Error::Serialization(message) => {
                    // check if the buffer actually contains TensorMap and
                    // not Labels
                    let slice = std::slice::from_raw_parts(buffer.cast::<u8>(), buffer_count);
                    let mut cursor = std::io::Cursor::new(slice);

                    if crate::io::looks_like_tensormap_data(crate::io::PathOrBuffer::Buffer(&mut cursor)) {
                        Error::Serialization(format!(
                            "unable to load Labels from buffer, use `load_buffer` to load TensorMap: {}", message
                        ))
                    } else if crate::io::looks_like_block_data(crate::io::PathOrBuffer::Buffer(&mut cursor)) {
                        Error::Serialization(format!(
                            "unable to load Labels from buffer, use `load_block_buffer` to load TensorBlock: {}", message
                        ))
                    } else {
                        Error::Serialization(format!(
                            "unable to load Labels from buffer: {}", message
                        ))
                    }
                }
                err => return err,
            })?;

        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = mts_labels_t::into_raw(Arc::new(rust_labels));

        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null();
    }

    return result;
}

/// Save labels to the file at the given path.
///
/// If the file already exists, it is overwritten. The recomended file extension
/// when saving data is `.mts`, to prevent confusion with generic `.npz` files.
///
/// @param path path to the file as a NULL-terminated UTF-8 string
/// @param labels pointer to labels to save to the file
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern "C" fn mts_labels_save(
    path: *const c_char,
    labels: *const mts_labels_t,
) -> mts_status_t {
    catch_unwind(move || {
        check_pointers_non_null!(path, labels);

        let path = CStr::from_ptr(path).to_str().expect("use UTF-8 for path");
        let mut file = BufWriter::new(File::create(path)?);

        let labels_ref: &crate::Labels = &*labels;
        crate::io::save_labels(&mut file, labels_ref)?;

        Ok(())
    })
}


/// Save labels to an in-memory buffer.
///
/// On input, `*buffer` should contain the address of a starting buffer (which
/// can be NULL) and `*buffer_count` should contain the size of the allocation
/// (0 if `*buffer` is NULL).
///
/// On output, `*buffer` will contain the serialized data, and `*buffer_count`
/// the total number of written bytes (which might be less than the allocation
/// size).
///
/// Users of this function are responsible for freeing the `*buffer` when they
/// are done with it, using the function matching the `realloc` callback.
///
/// @param buffer pointer to the buffer the tensor will be stored to, which can
///        change due to reallocations.
/// @param buffer_count pointer to the buffer size on input, number of written
///        bytes on output
/// @param realloc_user_data custom data for the `realloc` callback. This will
///        be passed as the first argument to `realloc` as-is.
/// @param realloc function that allows to grow the buffer allocation
/// @param labels pointer to labels that will saved to the buffer
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full error
///          message.
#[no_mangle]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern "C" fn mts_labels_save_buffer(
    buffer: *mut *mut u8,
    buffer_count: *mut usize,
    realloc_user_data: *mut c_void,
    realloc: mts_realloc_buffer_t,
    labels: *const mts_labels_t,
) -> mts_status_t {
    catch_unwind(move || {
        check_pointers_non_null!(labels);

        if realloc.is_none() {
            return Err(Error::InvalidParameter(
                "realloc callback can not be NULL in mts_labels_save_buffer".into()
            ));
        }

        if (*buffer).is_null() {
            assert_eq!(*buffer_count, 0);
        }

        let mut external_buffer = ExternalBuffer {
            data: buffer,
            allocated: *buffer_count as u64,
            writen: 0,
            realloc_user_data,
            realloc: realloc.expect("we checked"),
            current: 0,
        };

        let labels_ref: &crate::Labels = &*labels;
        crate::io::save_labels(&mut external_buffer, labels_ref)?;

        *buffer_count = external_buffer.current as usize;

        Ok(())
    })
}
