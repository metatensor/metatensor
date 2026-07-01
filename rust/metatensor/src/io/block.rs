use std::ffi::CString;

use dlpk::DLDataType;

use crate::errors::{check_ptr, check_status};
use crate::{Error, MtsArray, TensorBlock, TensorBlockRef};

use super::{realloc_vec, create_ndarray, CREATE_ARRAY_CALLBACK, create_array_callback_wrapper};

/// Load previously saved `TensorBlock` from the file at the given path.
///
/// All arrays are loaded as `ndarray::ArrayD` by default. If you want to load
/// arrays as a custom type, use [`load_block_custom_array`] instead.
pub fn load_block(path: impl AsRef<std::path::Path>) -> Result<TensorBlock, Error> {
    return load_block_custom_array(path, create_ndarray);
}

/// Load previously saved `TensorBlock` from the file at the given path.
///
/// All arrays will be created through the `create_array` callback.
pub fn load_block_custom_array<F>(path: impl AsRef<std::path::Path>, create_array: F) -> Result<TensorBlock, Error>
    where F: Fn(&[usize], DLDataType) -> Result<MtsArray, Error> + 'static
{
    let path = path.as_ref().as_os_str().to_str().expect("this path is not valid UTF8");
    let path = CString::new(path).expect("this path contains a NULL byte");

    CREATE_ARRAY_CALLBACK.with(|callback| {
        *callback.borrow_mut() = Some(Box::new(create_array));
    });

    let ptr = unsafe {
        crate::c_api::mts_block_load(
            path.as_ptr(),
            Some(create_array_callback_wrapper)
        )
    };

    CREATE_ARRAY_CALLBACK.with(|callback| {
        *callback.borrow_mut() = None;
    });

    check_ptr(ptr)?;

    return Ok(unsafe { TensorBlock::from_raw(ptr) });
}

/// Load a serialized `TensorBlock` from a `buffer`.
///
/// All arrays are loaded as `ndarray::ArrayD` by default. If you want to load
/// arrays as a custom type, use [`load_block_buffer_custom_array`] instead.
pub fn load_block_buffer(buffer: &[u8]) -> Result<TensorBlock, Error> {
    return load_block_buffer_custom_array(buffer, create_ndarray);
}

/// Load a serialized `TensorBlock` from a `buffer`.
///
/// All arrays will be created through the `create_array` callback.
pub fn load_block_buffer_custom_array<F>(buffer: &[u8], create_array: F) -> Result<TensorBlock, Error>
    where F: Fn(&[usize], DLDataType) -> Result<MtsArray, Error> + 'static
{
    CREATE_ARRAY_CALLBACK.with(|callback| {
        *callback.borrow_mut() = Some(Box::new(create_array));
    });

    let ptr = unsafe {
        crate::c_api::mts_block_load_buffer(
            buffer.as_ptr(),
            buffer.len(),
            Some(create_array_callback_wrapper)
        )
    };

    CREATE_ARRAY_CALLBACK.with(|callback| {
        *callback.borrow_mut() = None;
    });

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
            (buffer as *mut Vec<u8>).cast(),
            Some(realloc_vec),
            block.as_ptr(),
        ))?;
    }

    buffer.resize(buffer_count, 0);

    Ok(())
}
