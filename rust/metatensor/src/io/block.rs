use std::ffi::CString;

use metatensor_sys::mts_create_array_callback_t;

use crate::errors::{check_ptr, check_status};
use crate::{Error, TensorBlock, TensorBlockRef};

use super::{realloc_vec, create_ndarray};

/// Load previously saved `TensorBlock` from the file at the given path.
///
/// All arrays are loaded as `ndarray::ArrayD` by default. If you want to load
/// arrays as a custom type, use [`load_block_custom_array`] instead.
pub fn load_block(path: impl AsRef<std::path::Path>) -> Result<TensorBlock, Error> {
    return load_block_custom_array(path, Some(create_ndarray));
}

/// Load previously saved `TensorBlock` from the file at the given path.
///
/// All arrays will be created through the `create_array` callback.
pub fn load_block_custom_array(path: impl AsRef<std::path::Path>, create_array: mts_create_array_callback_t) -> Result<TensorBlock, Error> {
    let path = path.as_ref().as_os_str().to_str().expect("this path is not valid UTF8");
    let path = CString::new(path).expect("this path contains a NULL byte");

    let ptr = unsafe {
        crate::c_api::mts_block_load(
            path.as_ptr(),
            create_array
        )
    };

    check_ptr(ptr)?;

    return Ok(unsafe { TensorBlock::from_raw(ptr) });
}

/// Load a serialized `TensorBlock` from a `buffer`.
///
/// All arrays are loaded as `ndarray::ArrayD` by default. If you want to load
/// arrays as a custom type, use [`load_block_buffer_custom_array`] instead.
pub fn load_block_buffer(buffer: &[u8]) -> Result<TensorBlock, Error> {
    return load_block_buffer_custom_array(buffer, Some(create_ndarray));
}

/// Load a serialized `TensorBlock` from a `buffer`.
///
/// All arrays will be created through the `create_array` callback.
pub fn load_block_buffer_custom_array(buffer: &[u8], create_array: mts_create_array_callback_t) -> Result<TensorBlock, Error> {
    let ptr = unsafe {
        crate::c_api::mts_block_load_buffer(
            buffer.as_ptr(),
            buffer.len(),
            create_array
        )
    };

    check_ptr(ptr)?;

    return Ok(unsafe { TensorBlock::from_raw(ptr) });
}

/// Save the given `block` to a file.
///
/// If the file already exists, it is overwritten. The recommended file extension
/// when saving data is `.mts`, to prevent confusion with generic `.npz`.
pub fn save_block(path: impl AsRef<std::path::Path>, block: TensorBlockRef) -> Result<(), Error> {
    let path = path.as_ref().as_os_str().to_str().expect("this path is not valid UTF8");
    let path = CString::new(path).expect("this path contains a NULL byte");

    unsafe {
        check_status(crate::c_api::mts_block_save(path.as_ptr(), block.as_ptr()))
    }
}


/// Save the given `block` to an in-memory `buffer`.
///
/// This function will grow the buffer as required to fit the data.
pub fn save_block_buffer(block: TensorBlockRef, buffer: &mut Vec<u8>) -> Result<(), Error> {
    let mut buffer_ptr = buffer.as_mut_ptr();
    let mut buffer_count = buffer.len();

    unsafe {
        check_status(crate::c_api::mts_block_save_buffer(
            &mut buffer_ptr,
            &mut buffer_count,
            std::ptr::from_mut(buffer).cast(),
            Some(realloc_vec),
            block.as_ptr(),
        ))?;
    }

    buffer.resize(buffer_count, 0);

    Ok(())
}
