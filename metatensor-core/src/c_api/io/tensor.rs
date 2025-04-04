use std::os::raw::{c_char, c_void};
use std::ffi::CStr;
use std::fs::File;
use std::io::{BufReader, BufWriter};

use crate::Error;
use crate::data::mts_array_t;

use super::{ExternalBuffer, mts_realloc_buffer_t};

use super::super::status::{mts_status_t, catch_unwind};
use super::super::tensor::mts_tensormap_t;
use super::mts_create_array_callback_t;

/// Load a tensor map from the file at the given path.
///
/// Arrays for the values and gradient data will be created with the given
/// `create_array` callback, and filled by this function with the corresponding
/// data.
///
/// The memory allocated by this function should be released using
/// `mts_tensormap_free`.
///
/// `TensorMap` are serialized using numpy's NPZ format, i.e. a ZIP file
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
///          case of error. In case of error, you can use `mts_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern fn mts_tensormap_load(
    path: *const c_char,
    create_array: mts_create_array_callback_t,
) -> *mut mts_tensormap_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        check_pointers_non_null!(path);

        let create_array = wrap_create_array(&create_array);

        let path = CStr::from_ptr(path).to_str().expect("use UTF-8 for path");
        let file = BufReader::new(File::open(path)?);
        let tensor = crate::io::load(file, create_array)
            .map_err(|err| match err {
                Error::Serialization(message) => {
                    if crate::io::looks_like_labels_data(crate::io::PathOrBuffer::Path(path)) {
                        Error::Serialization(format!(
                            "unable to load a TensorMap from '{}', use `load_labels` to load Labels: {}", path, message
                        ))
                    } else if crate::io::looks_like_block_data(crate::io::PathOrBuffer::Path(path)) {
                        Error::Serialization(format!(
                            "unable to load a TensorMap from '{}', use `load_block` to load TensorBlock: {}", path, message
                        ))
                    } else {
                        Error::Serialization(format!(
                            "unable to load a TensorMap from '{}': {}", path, message
                        ))
                    }
                }
                err => return err,
            })?;

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *(unwind_wrapper.0) = mts_tensormap_t::into_boxed_raw(tensor);
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
/// `mts_tensormap_free`.
///
/// @param buffer buffer containing a previously serialized `mts_tensormap_t`
/// @param buffer_count number of elements in the buffer
/// @param create_array callback function that will be used to create data
///                     arrays inside each block
///
/// @returns A pointer to the newly allocated tensor map, or a `NULL` pointer in
///          case of error. In case of error, you can use `mts_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern fn mts_tensormap_load_buffer(
    buffer: *const u8,
    buffer_count: usize,
    create_array: mts_create_array_callback_t,
) -> *mut mts_tensormap_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        check_pointers_non_null!(buffer);
        assert!(buffer_count > 0);

        let create_array = wrap_create_array(&create_array);

        let slice = std::slice::from_raw_parts(buffer.cast::<u8>(), buffer_count);
        let cursor = std::io::Cursor::new(slice);

        let tensor = crate::io::load(cursor, create_array)
            .map_err(|err| match err {
                Error::Serialization(message) => {
                    let slice = std::slice::from_raw_parts(buffer.cast::<u8>(), buffer_count);
                    let mut cursor = std::io::Cursor::new(slice);
                    if crate::io::looks_like_labels_data(crate::io::PathOrBuffer::Buffer(&mut cursor)) {
                        Error::Serialization(format!(
                            "unable to load a TensorMap from buffer, use `load_labels_buffer` to load Labels: {}", message
                        ))
                    } else if crate::io::looks_like_block_data(crate::io::PathOrBuffer::Buffer(&mut cursor)) {
                        Error::Serialization(format!(
                            "unable to load a TensorMap from buffer, use `load_block_buffer` to load TensorBlock: {}", message
                        ))
                    } else {
                        Error::Serialization(format!(
                            "unable to load a TensorMap from buffer: {}", message
                        ))
                    }
                }
                err => return err,
            })?;

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *(unwind_wrapper.0) = mts_tensormap_t::into_boxed_raw(tensor);
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
                context: "failed to create a new array in mts_tensormap_load".into()
            });
        }
    }
}


/// Save a tensor map to the file at the given path.
///
/// If the file already exists, it is overwritten. The recomended file extension
/// when saving data is `.mts`, to prevent confusion with generic `.npz` files.
///
/// @param path path to the file as a NULL-terminated UTF-8 string
/// @param tensor tensor map to save to the file
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn mts_tensormap_save(
    path: *const c_char,
    tensor: *const mts_tensormap_t,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(path, tensor);

        let path = CStr::from_ptr(path).to_str().expect("use UTF-8 for path");
        let file = BufWriter::new(File::create(path)?);
        crate::io::save(file, &*tensor)?;

        Ok(())
    })
}


/// Save a tensor map to an in-memory buffer.
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
/// @param buffer pointer to the buffer the tensor will be stored to, which can
///        change due to reallocations.
/// @param buffer_count pointer to the buffer size on input, number of written
///        bytes on output
/// @param realloc_user_data custom data for the `realloc` callback. This will
///        be passed as the first argument to `realloc` as-is.
/// @param realloc function that allows to grow the buffer allocation
/// @param tensor tensor map that will saved to the buffer
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full error
///          message.
#[no_mangle]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern fn mts_tensormap_save_buffer(
    buffer: *mut *mut u8,
    buffer_count: *mut usize,
    realloc_user_data: *mut c_void,
    realloc: mts_realloc_buffer_t,
    tensor: *const mts_tensormap_t,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(tensor, buffer_count, buffer);

        if realloc.is_none() {
            return Err(Error::InvalidParameter(
                "realloc callback can not be NULL in mts_tensormap_save_buffer".into()
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

        crate::io::save(&mut external_buffer, &*tensor)?;

        *buffer_count = external_buffer.current as usize;

        Ok(())
    })
}
