use std::os::raw::{c_char, c_void};
use std::ffi::CStr;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use crate::Error;
use crate::data::mts_array_t;

use super::{ExternalBuffer, mts_realloc_buffer_t};

use super::super::status::{mts_status_t, catch_unwind};
use super::super::blocks::mts_block_t;
use super::mts_create_array_callback_t;


/// Load a tensor block from the file at the given path.
///
/// Arrays for the values and gradient data will be created with the given
/// `create_array` callback, and filled by this function with the corresponding
/// data.
///
/// The memory allocated by this function should be released using
/// `mts_block_free`.
///
/// See `mts_tensormap_load` for more information about the format used to
/// serialize the data.
///
/// @param path path to the file as a NULL-terminated UTF-8 string
/// @param create_array callback function that will be used to create data
///                     arrays inside each block
///
/// @returns A pointer to the newly allocated block, or a `NULL` pointer in
///          case of error. In case of error, you can use `mts_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern fn mts_block_load(
    path: *const c_char,
    create_array: mts_create_array_callback_t,
) -> *mut mts_block_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        check_pointers_non_null!(path);

        let create_array = wrap_create_array(&create_array);

        let path = CStr::from_ptr(path).to_str().expect("use UTF-8 for path");
        let file = BufReader::new(File::open(path)?);
        let block = crate::io::load_block(file, create_array)
            .map_err(|err| match err {
                Error::Serialization(message) => {
                    if crate::io::looks_like_labels_data(crate::io::PathOrBuffer::Path(path)) {
                        Error::Serialization(format!(
                            "unable to load a TensorBlock from '{}', use `load_labels` to load Labels: {}", path, message
                        ))
                    } else if crate::io::looks_like_tensormap_data(crate::io::PathOrBuffer::Path(path)) {
                        Error::Serialization(format!(
                            "unable to load a TensorBlock from '{}', use `load` to load TensorMap: {}", path, message
                        ))
                    } else {
                        Error::Serialization(format!(
                            "unable to load a TensorBlock from '{}': {}", path, message
                        ))
                    }
                }
                err => return err,
            })?;

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *(unwind_wrapper.0) = mts_block_t::into_boxed_raw(block);
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}

/// Load a tensor block from the given in-memory buffer.
///
/// Arrays for the values and gradient data will be created with the given
/// `create_array` callback, and filled by this function with the corresponding
/// data.
///
/// The memory allocated by this function should be released using
/// `mts_block_free`.
///
/// @param buffer buffer containing a previously serialized `mts_block_t`
/// @param buffer_count number of elements in the buffer
/// @param create_array callback function that will be used to create data
///                     arrays inside each block
///
/// @returns A pointer to the newly allocated tensor block, or a `NULL` pointer
///          in case of error. In case of error, you can use `mts_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern fn mts_block_load_buffer(
    buffer: *const u8,
    buffer_count: usize,
    create_array: mts_create_array_callback_t,
) -> *mut mts_block_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        check_pointers_non_null!(buffer);
        assert!(buffer_count > 0);

        let create_array = wrap_create_array(&create_array);

        let slice = std::slice::from_raw_parts(buffer.cast::<u8>(), buffer_count);
        let cursor = std::io::Cursor::new(slice);

        let block = crate::io::load_block(cursor, create_array)
            .map_err(|err| match err {
                Error::Serialization(message) => {
                    let slice = std::slice::from_raw_parts(buffer.cast::<u8>(), buffer_count);
                    let mut cursor = std::io::Cursor::new(slice);
                    if crate::io::looks_like_labels_data(crate::io::PathOrBuffer::Buffer(&mut cursor)) {
                        Error::Serialization(format!(
                            "unable to load a TensorBlock from buffer, use `load_labels` to load Labels: {}", message
                        ))
                    } else if crate::io::looks_like_tensormap_data(crate::io::PathOrBuffer::Buffer(&mut cursor)) {
                        Error::Serialization(format!(
                            "unable to load a TensorBlock from buffer, use `load` to load TensorMap: {}", message
                        ))
                    } else {
                        Error::Serialization(format!(
                            "unable to load a TensorBlock from buffer: {}", message
                        ))
                    }
                }
                err => return err,
            })?;

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *(unwind_wrapper.0) = mts_block_t::into_boxed_raw(block);
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}

fn wrap_create_array(create_array: &mts_create_array_callback_t) -> impl Fn(Vec<usize>) -> Result<mts_array_t, Error> + '_ {
    |shape: Vec<usize>| {
        let mut array = mts_array_t::null();
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
                context: "failed to create a new array in mts_block_load".into()
            });
        }
    }
}


/// Save a tensor block to the file at the given path.
///
/// If the file already exists, it is overwritten.
///
/// @param path path to the file as a NULL-terminated UTF-8 string
/// @param block tensor block to save to the file
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn mts_block_save(
    path: *const c_char,
    block: *const mts_block_t,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(path, block);

        let path = CStr::from_ptr(path).to_str().expect("use UTF-8 for path");
        let file = BufWriter::new(File::create(path)?);
        crate::io::save_block(file, &*block)?;

        Ok(())
    })
}


/// Save a tensor block to an in-memory buffer.
///
/// On input, `*buffer` should contain the address of a starting buffer (which
/// can be NULL) and `*buffer_count` should contain the size of the allocation.
///
/// On output, `*buffer` will contain the serialized data, and `*buffer_count`
/// the total number of written bytes (which might be less than the allocation
/// size).
///
/// Users of this function are responsible for freeing the `*buffer` when they
/// are done with it, using the function matching the `realloc` callback.
///
/// @param buffer pointer to the buffer the block will be stored to, which can
///        change due to reallocations.
/// @param buffer_count pointer to the buffer size on input, number of written
///        bytes on output
/// @param realloc_user_data custom data for the `realloc` callback. This will
///        be passed as the first argument to `realloc` as-is.
/// @param realloc function that allows to grow the buffer allocation
/// @param block tensor block that will saved to the buffer
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full error
///          message.
#[no_mangle]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern fn mts_block_save_buffer(
    buffer: *mut *mut u8,
    buffer_count: *mut usize,
    realloc_user_data: *mut c_void,
    realloc: mts_realloc_buffer_t,
    block: *const mts_block_t,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(block, buffer_count, buffer);

        if realloc.is_none() {
            return Err(Error::InvalidParameter(
                "realloc callback can not be NULL in mts_block_save_buffer".into()
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

        crate::io::save_block(&mut external_buffer, &*block)?;

        *buffer_count = external_buffer.current as usize;

        Ok(())
    })
}
