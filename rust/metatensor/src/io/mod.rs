//! Input/Output facilities for storing [`crate::TensorMap`] and
//! [`crate::Labels`] on disk

use std::os::raw::c_void;

use dlpk::sys::{DLDataType, DLDataTypeCode};

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

        Ok(())
    });

    if status != MTS_SUCCESS {
        return std::ptr::null_mut();
    }

    return result;
}

/// Create a typed `ndarray::ArcArray<T>` and box it as `dyn Array`.
macro_rules! create_typed_array {
    ($shape:expr, $c_array:expr, $T:ty) => {{
        let array = ndarray::ArcArray::<$T, _>::from_elem($shape, <$T>::default());
        *$c_array = (Box::new(array) as Box<dyn Array>).into();
    }};
}

/// callback used to create `ndarray::ArcArray` when loading a `TensorMap`
unsafe extern "C" fn create_ndarray(
    shape_ptr: *const usize,
    shape_count: usize,
    dtype: DLDataType,
    c_array: *mut mts_array_t,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        assert!(shape_count != 0);
        let shape = std::slice::from_raw_parts(shape_ptr, shape_count);

        if dtype.lanes != 1 {
            return Err(crate::Error {
                code: None,
                message: format!(
                    "unsupported dtype in create_ndarray: lanes={} (expected 1)",
                    dtype.lanes
                ),
            });
        }

        match (dtype.code, dtype.bits) {
            (DLDataTypeCode::kDLFloat, 32) => create_typed_array!(shape, c_array, f32),
            (DLDataTypeCode::kDLFloat, 64) => create_typed_array!(shape, c_array, f64),
            (DLDataTypeCode::kDLInt, 8) => create_typed_array!(shape, c_array, i8),
            (DLDataTypeCode::kDLInt, 16) => create_typed_array!(shape, c_array, i16),
            (DLDataTypeCode::kDLInt, 32) => create_typed_array!(shape, c_array, i32),
            (DLDataTypeCode::kDLInt, 64) => create_typed_array!(shape, c_array, i64),
            (DLDataTypeCode::kDLUInt, 8) => create_typed_array!(shape, c_array, u8),
            (DLDataTypeCode::kDLUInt, 16) => create_typed_array!(shape, c_array, u16),
            (DLDataTypeCode::kDLUInt, 32) => create_typed_array!(shape, c_array, u32),
            (DLDataTypeCode::kDLUInt, 64) => create_typed_array!(shape, c_array, u64),
            (DLDataTypeCode::kDLBool, 8) => create_typed_array!(shape, c_array, bool),
            (DLDataTypeCode::kDLFloat, 16) => create_typed_array!(shape, c_array, half::f16),
            (DLDataTypeCode::kDLComplex, 64) => create_typed_array!(shape, c_array, [f32; 2]),
            (DLDataTypeCode::kDLComplex, 128) => create_typed_array!(shape, c_array, [f64; 2]),
            _ => {
                return Err(crate::Error {
                    code: None,
                    message: format!(
                        "unsupported dtype in create_ndarray: code={:?} bits={}",
                        dtype.code, dtype.bits
                    ),
                });
            }
        }

        Ok(())
    })
}
