use std::os::raw::c_void;

mod labels;
mod tensor;

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
type mts_realloc_buffer_t = Option<unsafe extern fn(
    user_data: *mut c_void,
    ptr: *mut u8,
    new_size: usize
) -> *mut u8>;


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
