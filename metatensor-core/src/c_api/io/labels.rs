
use std::os::raw::{c_char, c_void};
use std::sync::Arc;
use std::ffi::CStr;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use dlpk::sys::{DLDataType, DLDataTypeCode};

use crate::Error;
use crate::Labels;
use crate::labels::LabelValue;
use crate::data::mts_array_t;

use super::{ExternalBuffer, mts_realloc_buffer_t, mts_create_array_callback_t, wrap_create_array};

use super::super::status::{mts_status_t, catch_unwind};
use super::super::labels::mts_labels_t;

/// Default array creation for `mts_labels_load` and `mts_labels_load_buffer`:
/// create an array backed by an owned `Vec<LabelValue>` with the given shape,
/// which is then filled by the loading code.
fn default_create_labels_array(shape: &[usize], dtype: DLDataType) -> Result<mts_array_t, Error> {
    if shape.len() != 2 || dtype.code != DLDataTypeCode::kDLInt || dtype.bits != 32 || dtype.lanes != 1 {
        return Err(Error::InvalidParameter(format!(
            "invalid shape or dtype for labels: shape={:?}, dtype={:?}",
            shape, dtype
        )));
    }

    let values = vec![0 as LabelValue; shape[0] * shape[1]];
    return Ok(crate::labels::create_array_from_vec(
        Arc::from(values.into_boxed_slice()),
        shape[0],
        shape[1],
    ));
}

/// Load labels from a file, creating the values array with the given
/// `create_array` closure and mapping errors to indicate whether the data
/// actually looks like TensorMap or TensorBlock.
fn load_labels_from_path<F>(path: &str, create_array: F) -> Result<Labels, Error>
where F: Fn(&[usize], DLDataType) -> Result<mts_array_t, Error>
{
    let file = BufReader::new(File::open(path)?);

    crate::io::load_labels(file, create_array).map_err(|err| match err {
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
        err => err,
    })
}

/// Load labels from an in-memory buffer, creating the values array with the
/// given `create_array` closure and mapping errors to indicate whether the
/// data actually looks like TensorMap or TensorBlock.
fn load_labels_from_buffer<F>(slice: &[u8], create_array: F) -> Result<Labels, Error>
where F: Fn(&[usize], DLDataType) -> Result<mts_array_t, Error>
{
    let cursor = std::io::Cursor::new(slice);

    crate::io::load_labels(cursor, create_array).map_err(|err| match err {
        Error::Serialization(message) => {
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
        err => err,
    })
}

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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mts_labels_load(path: *const c_char) -> *const mts_labels_t {
    labels_load_impl(path, default_create_labels_array)
}

/// Load labels from the file at the given path.
///
/// The array containing the labels values will be created with the given
/// `create_array` callback, and filled by this function with the
/// corresponding data.
///
/// This function allocates memory which must be released with
/// `mts_labels_free` when you don't need it anymore.
///
/// @param path path to the file as a NULL-terminated UTF-8 string
/// @param create_array callback function that will be used to create the
///                     array containing the labels values.
///
/// @returns A pointer to the newly allocated labels, or a `NULL` pointer in
///          case of error. In case of error, you can use `mts_last_error()`
///          to get the error message.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mts_labels_load_custom_array(
    path: *const c_char,
    create_array: mts_create_array_callback_t,
) -> *const mts_labels_t {
    let create_array = wrap_create_array(&create_array);
    labels_load_impl(path, move |shape, dtype| create_array(shape, dtype))
}

/// Implementation shared by `mts_labels_load` and `mts_labels_load_custom_array`.
fn labels_load_impl<F>(path: *const c_char, create_array: F) -> *const mts_labels_t
where F: Fn(&[usize], DLDataType) -> Result<mts_array_t, Error> + std::panic::UnwindSafe
{
    let mut result = std::ptr::null();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);

    let status = catch_unwind(move || {
        check_pointers_non_null!(path);

        let path = unsafe {
            CStr::from_ptr(path).to_str().expect("use UTF-8 for path")
        };

        let rust_labels = load_labels_from_path(path, create_array)?;

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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mts_labels_load_buffer(
    buffer: *const u8,
    buffer_count: usize,
) -> *const mts_labels_t {
    labels_load_buffer_impl(buffer, buffer_count, default_create_labels_array)
}

/// Load labels from the given in-memory buffer.
///
/// The array containing the labels values will be created with the given
/// `create_array` callback, and filled by this function with the
/// corresponding data.
///
/// This function allocates memory which must be released with
/// `mts_labels_free` when you don't need it anymore.
///
/// @param buffer buffer containing a previously serialized `mts_labels_t`
/// @param buffer_count number of elements in the buffer
/// @param create_array callback function that will be used to create the
///                     array containing the labels values.
///
/// @returns A pointer to the newly allocated labels, or a `NULL` pointer in
///          case of error. In case of error, you can use `mts_last_error()`
///          to get the error message.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mts_labels_load_buffer_custom_array(
    buffer: *const u8,
    buffer_count: usize,
    create_array: mts_create_array_callback_t,
) -> *const mts_labels_t {
    let create_array = wrap_create_array(&create_array);
    labels_load_buffer_impl(buffer, buffer_count, move |shape, dtype| create_array(shape, dtype))
}

/// Implementation shared by `mts_labels_load_buffer` and
/// `mts_labels_load_buffer_custom_array`.
fn labels_load_buffer_impl<F>(buffer: *const u8, buffer_count: usize, create_array: F) -> *const mts_labels_t
where F: Fn(&[usize], DLDataType) -> Result<mts_array_t, Error> + std::panic::UnwindSafe
{
    let mut result = std::ptr::null();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);

    let status = catch_unwind(move || {
        if buffer_count == 0 {
            return Err(Error::Serialization("unable to load Labels from empty buffer".into()));
        }

        check_pointers_non_null!(buffer);

        let slice = unsafe {
            std::slice::from_raw_parts(buffer.cast::<u8>(), buffer_count)
        };

        let rust_labels = load_labels_from_buffer(slice, create_array)?;

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
/// If the file already exists, it is overwritten. The recommended file extension
/// when saving data is `.mts`, to prevent confusion with generic `.npz` files.
///
/// @param path path to the file as a NULL-terminated UTF-8 string
/// @param labels pointer to labels to save to the file
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mts_labels_save(
    path: *const c_char,
    labels: *const mts_labels_t,
) -> mts_status_t {
    catch_unwind(move || {
        check_pointers_non_null!(path, labels);

        let path = unsafe {
            CStr::from_ptr(path).to_str().expect("use UTF-8 for path")
        };
        let mut file = BufWriter::new(File::create(path)?);

        let labels = unsafe { &*labels };
        crate::io::save_labels(&mut file, labels)?;

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
/// @param labels pointer to labels that will be saved to the buffer
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full error
///          message.
#[unsafe(no_mangle)]
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

        if unsafe { (*buffer).is_null() && (*buffer_count) != 0 } {
            return Err(Error::InvalidParameter(
                "`buffer` is NULL but `buffer_count` is not 0 in mts_labels_save_buffer".into()
            ));
        }

        let mut external_buffer = ExternalBuffer {
            data: buffer,
            allocated: unsafe { *buffer_count } as u64,
            writen: 0,
            realloc_user_data,
            realloc: realloc.expect("we checked"),
            current: 0,
        };

        let labels = unsafe { &*labels };
        crate::io::save_labels(&mut external_buffer, labels)?;

        unsafe {
            *buffer_count = external_buffer.current as usize;
        }

        Ok(())
    })
}
