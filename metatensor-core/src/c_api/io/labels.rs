
use std::os::raw::{c_char, c_void};
use std::sync::Arc;
use std::ffi::CStr;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use crate::Error;

use super::{ExternalBuffer, mts_realloc_buffer_t};

use super::super::status::{mts_status_t, catch_unwind};
use super::super::labels::{mts_labels_t, rust_to_mts_labels, mts_labels_to_rust};

/// Load labels from the file at the given path.
///
/// Labels are stored using numpy's NPY format, so the file will typically use
/// the `.npy` extension.
///
/// This function allocates memory which must be released `mts_labels_free` when
/// you don't need it anymore.
///
/// @param path path to the file as a NULL-terminated UTF-8 string
/// @param labels pointer to empty Labels
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn mts_labels_load(
    path: *const c_char,
    labels: *mut mts_labels_t,
) -> mts_status_t {
    catch_unwind(move || {
        check_pointers_non_null!(path, labels);
        if (*labels).is_rust() {
            return Err(Error::InvalidParameter(
                "these labels already correspond to rust labels".into()
            ));
        }

        let path = CStr::from_ptr(path).to_str().expect("use UTF-8 for path");
        let file = BufReader::new(File::open(path)?);

        let rust_labels = crate::io::load_labels(file)
            .map_err(|err| match err {
                Error::Serialization(message) => {
                    if crate::io::looks_like_tensormap_data(crate::io::PathOrBuffer::Path(path)) {
                    Error::Serialization(format!(
                        "unable to load Labels from '{}', use `load` to load TensorMap: {}", path, message
                    ))
                } else {
                    Error::Serialization(format!(
                        "unable to load Labels from '{}': {}", path, message
                    ))
                }
                }
                err => return err,
            })?;

        *labels = rust_to_mts_labels(Arc::new(rust_labels));

        Ok(())
    })
}

/// Load labels from the given in-memory buffer.
///
/// This function allocates memory which must be released `mts_labels_free` when
/// you don't need it anymore.
///
/// @param buffer buffer containing a previously serialized `mts_labels_t`
/// @param buffer_count number of elements in the buffer
/// @param labels pointer to empty Labels
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn mts_labels_load_buffer(
    buffer: *const u8,
    buffer_count: usize,
    labels: *mut mts_labels_t,
) -> mts_status_t {
    catch_unwind(move || {
        check_pointers_non_null!(buffer, labels);
        if (*labels).is_rust() {
            return Err(Error::InvalidParameter(
                "these labels already correspond to rust labels".into()
            ));
        }

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
                    } else {
                        Error::Serialization(format!(
                            "unable to load Labels from buffer: {}", message
                        ))
                    }
                }
                err => return err,
            })?;

        *labels = rust_to_mts_labels(Arc::new(rust_labels));

        Ok(())
    })
}

/// Save labels to the file at the given path.
///
/// If the file already exists, it is overwritten.
///
/// @param path path to the file as a NULL-terminated UTF-8 string
/// @param labels Labels to save to the file
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn mts_labels_save(
    path: *const c_char,
    labels: mts_labels_t,
) -> mts_status_t {
    catch_unwind(move || {
        check_pointers_non_null!(path);
        if !labels.is_rust() {
            return Err(Error::InvalidParameter(
                "these labels do not support calling mts_labels_save, \
                call mts_labels_create first".into()
            ));
        }

        let path = CStr::from_ptr(path).to_str().expect("use UTF-8 for path");
        let mut file = BufWriter::new(File::create(path)?);

        let labels = mts_labels_to_rust(&labels)?;
        crate::io::save_labels(&mut file, &labels)?;

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
/// @param labels Labels that will saved to the buffer
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full error
///          message.
#[no_mangle]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern fn mts_labels_save_buffer(
    buffer: *mut *mut u8,
    buffer_count: *mut usize,
    realloc_user_data: *mut c_void,
    realloc: mts_realloc_buffer_t,
    labels: mts_labels_t,
) -> mts_status_t {
    catch_unwind(move || {
        if !labels.is_rust() {
            return Err(Error::InvalidParameter(
                "these labels do not support calling mts_labels_save_buffer, \
                call mts_labels_create first".into()
            ));
        }

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
            len: *buffer_count,
            realloc_user_data,
            realloc: realloc.expect("we checked"),
            current: 0,
        };

        let labels = mts_labels_to_rust(&labels)?;
        crate::io::save_labels(&mut external_buffer, &labels)?;

        *buffer_count = external_buffer.current as usize;

        Ok(())
    })
}
