//! Input/Output facilities for storing [`TensorMap`] on disk

use std::ffi::CString;
use std::os::raw::c_void;

use crate::c_api::{eqs_array_t, eqs_status_t, EQS_SUCCESS};
use crate::errors::{check_status, check_ptr};
use crate::{TensorMap, Error, Array};

/// Load the serialized tensor map from the given path.
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
pub fn load(path: impl AsRef<std::path::Path>) -> Result<TensorMap, Error> {
    let path = path.as_ref().as_os_str().to_str().expect("this path is not valid UTF8");
    let path = CString::new(path).expect("this path contains a NULL byte");

    let ptr = unsafe {
        crate::c_api::eqs_tensormap_load(
            path.as_ptr(),
            Some(create_ndarray)
        )
    };

    check_ptr(ptr)?;

    return Ok(unsafe { TensorMap::from_raw(ptr) });
}

/// Load a serialized `TensorMap` from a `buffer`.
///
/// See the [`load`] function for more information on the data format.
pub fn load_buffer(buffer: &[u8]) -> Result<TensorMap, Error> {
    let ptr = unsafe {
        crate::c_api::eqs_tensormap_load_buffer(
            buffer.as_ptr(),
            buffer.len(),
            Some(create_ndarray)
        )
    };

    check_ptr(ptr)?;

    return Ok(unsafe { TensorMap::from_raw(ptr) });
}

/// Save the given tensor to a file.
///
/// The format used is documented in the [`load`] function, and is based on
/// numpy's NPZ format (i.e. zip archive containing NPY files).
pub fn save(path: impl AsRef<std::path::Path>, tensor: &TensorMap) -> Result<(), Error> {
    let path = path.as_ref().as_os_str().to_str().expect("this path is not valid UTF8");
    let path = CString::new(path).expect("this path contains a NULL byte");

    unsafe {
        check_status(crate::c_api::eqs_tensormap_save(path.as_ptr(), tensor.ptr))
    }
}


/// Implementation of realloc for `Vec<u8>`, used in `save_buffer`
unsafe extern fn realloc_vec(user_data: *mut c_void, _ptr: *mut u8, new_size: usize) -> *mut u8 {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);

    let status = crate::errors::catch_unwind(move || {
        let vector = &mut *user_data.cast::<Vec<u8>>();
        vector.resize(new_size, 0);

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *(unwind_wrapper.0) = vector.as_mut_ptr();
    });

    if status != EQS_SUCCESS {
        return std::ptr::null_mut();
    }

    return result;
}

/// Save the given `tensor` to an in-memory `buffer`.
///
/// This function will grow the buffer as required to fit the whole tensor.
pub fn save_buffer(tensor: &TensorMap, buffer: &mut Vec<u8>) -> Result<(), Error> {
    let mut buffer_ptr = buffer.as_mut_ptr();
    let mut buffer_count = buffer.len();

    unsafe {
        check_status(crate::c_api::eqs_tensormap_save_buffer(
            &mut buffer_ptr,
            &mut buffer_count,
            (buffer as *mut Vec<u8>).cast(),
            Some(realloc_vec),
            tensor.ptr,
        ))?;
    }

    buffer.resize(buffer_count, 0);

    Ok(())
}

/// callback used to create `ndarray::ArrayD` when loading a `TensorMap`
unsafe extern fn create_ndarray(
    shape_ptr: *const usize,
    shape_count: usize,
    c_array: *mut eqs_array_t,
) -> eqs_status_t {
    crate::errors::catch_unwind(|| {
        assert!(shape_count != 0);
        let shape = std::slice::from_raw_parts(shape_ptr, shape_count);
        let array = ndarray::ArrayD::from_elem(shape, 0.0);
        *c_array = (Box::new(array) as Box<dyn Array>).into();
    })
}
