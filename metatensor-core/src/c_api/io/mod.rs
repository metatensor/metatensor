use std::os::raw::c_void;

use dlpk::sys::DLDataType;

use crate::data::mts_array_t;
use super::status::mts_status_t;

mod labels;
mod block;
mod tensor;

/// Function pointer to create a new `mts_array_t` when de-serializing tensor
/// maps.
///
/// This function gets the `shape` of the array (the `shape` contains
/// `shape_count` elements) and the `dtype` (a `DLDataType` describing the
/// element type), and should fill `array` with a new valid `mts_array_t` or
/// return non-zero `mts_status_t`.
///
/// The newly created array should live on CPU, since metatensor will use
/// `mts_array_t.as_dlpack` to get the data pointer and write to it.
///
/// This function should return `MTS_SUCCESS` on success, or
/// `MTS_CALLBACK_ERROR` on failure. In case of failure, the implementation
/// should call `mts_set_last_error` with an appropriate error message before
/// returning.
#[allow(non_camel_case_types)]
type mts_create_array_callback_t = unsafe extern "C" fn(
    shape: *const usize,
    shape_count: usize,
    dtype: DLDataType,
    array: *mut mts_array_t,
) -> mts_status_t;

/// Function pointer used by `mts_tensormap_load_mmap` /
/// `mts_block_load_mmap` to materialise each value/gradient array.
///
/// metatensor parses the NPY header for every array, then calls this
/// callback with the array's `shape`, DLPack `dtype`, and byte offset of
/// the raw data within the file. The implementation decides how to build
/// the resulting `mts_array_t`: it can wrap the corresponding mmap
/// region as a zero-copy view, copy the bytes into an owned buffer, or
/// upload them straight to a GPU via GPU Direct Storage.
///
/// `user_data` is the opaque pointer forwarded from
/// `mts_tensormap_load_mmap` / `mts_block_load_mmap`. It can carry a
/// cached file descriptor, an `mmap` handle, a GDS context, or anything
/// else the binding needs.
///
/// Byte length of the data region is always derivable from
/// `shape` and `dtype` (`product(shape) * dtype.bits / 8 * dtype.lanes`),
/// so it is not passed explicitly.
///
/// Returns `MTS_SUCCESS` on success or `MTS_CALLBACK_ERROR` on failure.
/// On failure, the implementation should call `mts_set_last_error` first.
#[allow(non_camel_case_types)]
pub(crate) type mts_create_file_array_callback_t = Option<unsafe extern "C" fn(
    user_data: *mut c_void,
    shape: *const usize,
    shape_count: usize,
    dtype: DLDataType,
    file_offset: usize,
    array: *mut mts_array_t,
) -> mts_status_t>;

/// Function pointer to grow in-memory buffers for `mts_tensormap_save_buffer`
/// and `mts_labels_save_buffer`.
///
/// This function takes an existing pointer in `ptr` and a new length in
/// `new_size`, and grow the allocation. If the pointer is `NULL`, it should
/// create a new allocation. If it is unable to allocate memory, it should
/// return a `NULL` pointer. This follows the API of the standard C function
/// `realloc`, with an additional parameter `user_data` that can be used to hold
/// custom data.
#[allow(non_camel_case_types)]
type mts_realloc_buffer_t = Option<unsafe extern "C" fn(
    user_data: *mut c_void,
    ptr: *mut u8,
    new_size: usize
) -> *mut u8>;


/// Wrapper for an externally managed buffer, that can be grown to fit more data
struct ExternalBuffer {
    data: *mut *mut u8,
    writen: u64,
    allocated: u64,

    realloc_user_data: *mut c_void,
    realloc: unsafe extern "C" fn(*mut c_void, *mut u8, usize) -> *mut u8,

    current: u64,
}

impl std::io::Write for ExternalBuffer {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let remaining_space = self.allocated.saturating_sub(self.current);

        if (remaining_space as usize) < buf.len() {
            let required_size = self.current.saturating_add(buf.len() as u64);
            let mut new_size = if self.allocated == 0 { 1024 } else { self.allocated };
            while new_size < required_size {
                new_size = new_size.saturating_mul(2);
                if new_size == 0 {
                     return Err(std::io::Error::new(
                         std::io::ErrorKind::OutOfMemory,
                         "requested allocation size overflow",
                     ));
                }
            }

            let new_ptr = unsafe {
                (self.realloc)(self.realloc_user_data, *self.data, new_size as usize)
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

            self.allocated = new_size;
        }

        let mut output = unsafe {
            let start = (*self.data).offset(self.current as isize);
            // allocated >= current + buf.len()
            std::slice::from_raw_parts_mut(start, buf.len())
        };

        let count = output.write(buf).expect("failed to write to pre-allocated slice");
        assert_eq!(count, buf.len());
        self.current += count as u64;

        if self.current > self.writen {
            self.writen = self.current;
        }
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
                if offset > self.writen {
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

                if -offset > self.writen as i64 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof, "tried to seek past the beginning of the buffer")
                    );
                }

                self.current = (self.writen as i64 + offset) as u64;
            },

            std::io::SeekFrom::Current(offset) => {
                let result = self.current as i64 + offset;
                if result > self.writen as i64 {
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
