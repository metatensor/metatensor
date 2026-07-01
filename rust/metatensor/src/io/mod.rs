//! Input/Output facilities for storing [`crate::TensorMap`] and
//! [`crate::Labels`] on disk

use std::cell::RefCell;
use std::os::raw::c_void;

use dlpk::sys::{DLDataType, DLDataTypeCode};

use crate::c_api::{MTS_SUCCESS, mts_array_t, mts_status_t};
use crate::MtsArray;

use crate::Error;
use crate::errors::catch_unwind;

mod tensor;
pub use self::tensor::{load, load_custom_array, load_buffer, load_buffer_custom_array};
pub use self::tensor::{save, save_buffer};

mod block;
pub use self::block::{load_block, load_block_buffer_custom_array, load_block_buffer, load_block_custom_array};
pub use self::block::{save_block, save_block_buffer};

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

/// Create a typed `ndarray::Array<T>` and put it in a `MtsArray`
macro_rules! create_typed_array {
    ($shape:expr, $c_array:expr, $T:ty) => {{
        let array = ndarray::Array::<$T, _>::from_elem($shape, <$T>::default());
        std::convert::Into::<MtsArray>::into(array)
    }};
}


fn create_ndarray(shape: &[usize], dtype: DLDataType) -> Result<MtsArray, Error> {
    if dtype.lanes != 1 {
        return Err(crate::Error {
            code: None,
            message: format!(
                "unsupported dtype in create_ndarray: lanes={} (expected 1)",
                dtype.lanes
            ),
        });
    }

    let array = match (dtype.code, dtype.bits) {
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
    };

    return Ok(array);
}

type CreateArrayCallback = dyn Fn(&[usize], DLDataType) -> Result<MtsArray, Error>;
thread_local! {
    static CREATE_ARRAY_CALLBACK: RefCell<Option<Box<CreateArrayCallback>>> = RefCell::new(None);
}

/// Wrap the function in `CREATE_ARRAY_CALLBACK` as a C-compatible callback
unsafe extern "C" fn create_array_callback_wrapper(
    shape: *const usize,
    shape_count: usize,
    dtype: DLDataType,
    array: *mut mts_array_t,
) -> mts_status_t {
    catch_unwind(|| {
        let shape = unsafe {
            std::slice::from_raw_parts(shape, shape_count)
        };

        let new_array = CREATE_ARRAY_CALLBACK.with(|callback| {
            let borrow = callback.borrow();
            let callback = borrow.as_ref().expect("no custom array callback set in thread-local storage");
            callback(shape, dtype)
        })?;

        unsafe {
            *array = new_array.into_raw();
        }

        Ok(())
    })
}
