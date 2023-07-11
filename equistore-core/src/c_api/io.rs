use std::os::raw::{c_char, c_void};
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


/// Wrapper for an externally managed buffer, that can be grown to fit more data
struct ExternalBuffer {
    data: *mut *mut u8,
    len: usize,

    realloc_user_data: *mut c_void,
    realloc: unsafe extern fn(*mut c_void, *mut u8, usize) -> *mut u8,

    current: u64,
}

impl std::io::Write for ExternalBuffer {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut remaining_space = self.len - self.current as usize;

        if remaining_space < buf.len() {
            // find the new size to be able to fit all the data
            let mut new_size = 0;
            while remaining_space < buf.len() {
                new_size = if self.len == 0 {
                    1024
                } else {
                    2 * self.len
                };
                remaining_space = new_size - self.current as usize;
            }

            let new_ptr = unsafe {
                (self.realloc)(self.realloc_user_data, *self.data, new_size)
            };

            if new_ptr.is_null() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "failed to allocate memory with the realloc callback"
                ));
            }

            unsafe {
                *self.data = new_ptr;
            }

            self.len = new_size;
        }

        let mut output = unsafe {
            let start = (*self.data).offset(self.current as isize);
            std::slice::from_raw_parts_mut(start, remaining_space)
        };

        let count = output.write(buf).expect("failed to write to pre-allocated slice");
        self.current += count as u64;
        return Ok(count);
    }

    fn flush(&mut self) -> std::io::Result<()> {
        return Ok(());
    }
}


#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
impl std::io::Seek for ExternalBuffer {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        match pos {
            std::io::SeekFrom::Start(offset) => {
                if offset > self.len as u64 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof, "tried to seek past the end of the buffer")
                    );
                }

                self.current = offset;
            },

            std::io::SeekFrom::End(offset) => {
                if offset > 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof, "tried to seek past the end of the buffer")
                    );
                }

                if -offset > self.len as i64 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof, "tried to seek past the beginning of the buffer")
                    );
                }

                self.current = (self.len as i64 + offset) as u64;
            },

            std::io::SeekFrom::Current(offset) => {
                let result = self.current as i64 + offset;
                if result > self.len as i64 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof, "tried to seek past the end of the buffer")
                    );
                }

                if result < 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof, "tried to seek past the beginning of the buffer")
                    );
                }

                self.current = result as u64;
            },
        }

        return Ok(self.current);
    }

    fn rewind(&mut self) -> std::io::Result<()> {
        self.current = 0;
        return Ok(());
    }

    fn stream_position(&mut self) -> std::io::Result<u64> {
        return Ok(self.current);
     }
}


/// Save a tensor map to an in-memory buffer.
///
/// The `realloc` callback should take an existing pointer and a new length, and
/// grow the allocation. If the pointer is `NULL`, it should create a new
/// allocation. If it is unable to allocate memory, it should return a `NULL`
/// pointer. This follows the API of the standard C function `realloc`, with an
/// additional parameter `user_data` that can be used to hold custom data.
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
/// @param realloc_user_data Custom data for the `realloc` callback. This will
///        be passed as the first argument to `realloc` as-is.
/// @param realloc function that allows to grow the buffer allocation
/// @param tensor tensor map that will saved to the buffer
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full error
///          message.
#[no_mangle]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern fn eqs_tensormap_save_buffer(
    buffer: *mut *mut u8,
    buffer_count: *mut usize,
    realloc_user_data: *mut c_void,
    realloc: Option<unsafe extern fn(user_data: *mut c_void, ptr: *mut u8, new_size: usize) -> *mut u8>,
    tensor: *const eqs_tensormap_t,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(tensor, buffer_count, buffer);

        if realloc.is_none() {
            return Err(Error::InvalidParameter(
                "realloc callback can not be NULL in eqs_tensormap_save_buffer".into()
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
