use std::ffi::CString;

use metatensor_sys::mts_create_array_callback_t;

use crate::errors::{check_status, check_ptr};
use crate::{Labels, Error};

use super::{realloc_vec, create_ndarray};

/// Load previously saved `Labels` from the file at the given path.
///
/// The labels values are loaded as `ndarray::ArrayD` by default. If you want
/// to load them as a custom type, use [`load_labels_custom_array`] instead.
pub fn load_labels(path: impl AsRef<std::path::Path>) -> Result<Labels, Error> {
    return load_labels_custom_array(path, Some(create_ndarray));
}

/// Load previously saved `Labels` from the file at the given path.
///
/// The array containing the labels values will be created through the
/// `create_array` callback.
pub fn load_labels_custom_array(path: impl AsRef<std::path::Path>, create_array: mts_create_array_callback_t) -> Result<Labels, Error> {
    let path = path.as_ref().as_os_str().to_str().expect("this path is not valid UTF8");
    let path = CString::new(path).expect("this path contains a NULL byte");

    let ptr = unsafe {
        crate::c_api::mts_labels_load_custom_array(
            path.as_ptr(),
            create_array,
        )
    };
    check_ptr(ptr)?;

    return Ok(unsafe { Labels::from_raw(ptr) });
}

/// Load previously saved `Labels` from an in-memory `buffer`.
///
/// The labels values are loaded as `ndarray::ArrayD` by default. If you want
/// to load them as a custom type, use [`load_labels_buffer_custom_array`]
/// instead.
pub fn load_labels_buffer(buffer: &[u8]) -> Result<Labels, Error> {
    return load_labels_buffer_custom_array(buffer, Some(create_ndarray));
}

/// Load previously saved `Labels` from an in-memory `buffer`.
///
/// The array containing the labels values will be created through the
/// `create_array` callback.
pub fn load_labels_buffer_custom_array(buffer: &[u8], create_array: mts_create_array_callback_t) -> Result<Labels, Error> {
    let ptr = unsafe {
        crate::c_api::mts_labels_load_buffer_custom_array(
            buffer.as_ptr(),
            buffer.len(),
            create_array,
        )
    };
    check_ptr(ptr)?;

    return Ok(unsafe { Labels::from_raw(ptr) });
}

/// Save the given `Labels` to a file.
///
/// If the file already exists, it is overwritten. The recommended file extension
/// when saving data is `.mts`, to prevent confusion with generic `.npz`.
pub fn save_labels(path: impl AsRef<std::path::Path>, labels: &Labels) -> Result<(), Error> {
    let path = path.as_ref().as_os_str().to_str().expect("this path is not valid UTF8");
    let path = CString::new(path).expect("this path contains a NULL byte");

    unsafe {
        check_status(crate::c_api::mts_labels_save(path.as_ptr(), labels.as_mts_labels_t()))
    }
}


/// Save the given `labels` to an in-memory `buffer`.
///
/// This function will grow the buffer as required to fit the labels.
pub fn save_labels_buffer(labels: &Labels, buffer: &mut Vec<u8>) -> Result<(), Error> {
    let mut buffer_ptr = buffer.as_mut_ptr();
    let mut buffer_count = buffer.len();

    unsafe {
        check_status(crate::c_api::mts_labels_save_buffer(
            &mut buffer_ptr,
            &mut buffer_count,
            std::ptr::from_mut(buffer).cast(),
            Some(realloc_vec),
            labels.as_mts_labels_t(),
        ))?;
    }

    buffer.resize(buffer_count, 0);

    Ok(())
}
