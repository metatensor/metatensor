//! Input/Output facilities for storing [`crate::TensorMap`] and
//! [`crate::Labels`] on disk

use std::os::raw::c_void;

use crate::c_api::{MTS_SUCCESS, mts_array_t, mts_status_t};
use crate::Array;

mod tensor;
pub use self::tensor::{load, save, load_buffer, save_buffer};

mod block;
pub use self::block::{load_block, load_block_buffer, save_block, save_block_buffer};

mod labels;
pub use self::labels::{load_labels, load_labels_buffer, save_labels, save_labels_buffer};


/// Implementation of realloc for `Vec<u8>`, used in `save_buffer`
unsafe extern "C" fn realloc_vec(user_data: *mut c_void, _ptr: *mut u8, new_size: usize) -> *mut u8 {
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

    if status != MTS_SUCCESS {
        return std::ptr::null_mut();
    }

    return result;
}

/// callback used to create `ndarray::ArrayD` when loading a `TensorMap`
unsafe extern "C" fn create_ndarray(
    shape_ptr: *const usize,
    shape_count: usize,
    c_array: *mut mts_array_t,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        assert!(shape_count != 0);
        let shape = std::slice::from_raw_parts(shape_ptr, shape_count);
        let array = ndarray::ArrayD::from_elem(shape, 0.0);
        *c_array = (Box::new(array) as Box<dyn Array>).into();
    })
}
