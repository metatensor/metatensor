//! Input/Output facilities for storing [`crate::TensorMap`] and
//! [`crate::Labels`] on disk

use std::os::raw::c_void;

use crate::c_api::{MTS_SUCCESS, mts_array_t, mts_status_t};
use crate::{Array, Error};

mod tensor;
pub use self::tensor::{load, save, load_buffer, save_buffer, load_mmap, load_partial};

mod block;
pub use self::block::{load_block, load_block_buffer, save_block, save_block_buffer, load_block_mmap};

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
        let array = ndarray::ArcArray::from_elem(shape, 0.0);
        *c_array = (Box::new(array) as Box<dyn Array>).into();
    })
}

/// callback used to create `ndarray::ArrayD` from mmap'ed data
unsafe extern "C" fn create_mmap_ndarray(
    shape_ptr: *const usize,
    shape_count: usize,
    dtype: dlpk::sys::DLDataType,
    data: *const std::os::raw::c_void,
    _data_len: usize,
    mmap_ptr: *mut std::os::raw::c_void,
    c_array: *mut mts_array_t,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        let shape = std::slice::from_raw_parts(shape_ptr, shape_count);

        // We wrap the raw pointer into a typed ArcArray via MmapNdarray.
        // The mmap lifetime is tied to the MmapNdarray wrapper.
        let array = MmapNdarray::new(mmap_ptr, data, shape.to_vec(), dtype);
        *c_array = (Box::new(array) as Box<dyn Array>).into();
    })
}

/// Create a typed `ArcArray<T>` from a raw mmap pointer.
///
/// If the pointer is aligned for `T`, a zero-copy shared view is created.
/// Otherwise the data is copied into an owned allocation.
unsafe fn mmap_array_typed<T>(data: *const c_void, shape: Vec<usize>) -> Box<dyn Array>
where
    T: 'static + Send + Sync + Clone + Default + dlpk::GetDLPackDataType,
{
    if data as usize % std::mem::align_of::<T>() == 0 {
        let view = ndarray::ArrayViewD::from_shape_ptr(shape, data.cast::<T>());
        let array = view.to_shared();
        Box::new(array)
    } else {
        // Not aligned: copy raw bytes to avoid alignment UB
        let len = shape.iter().product::<usize>();
        let mut vec = vec![T::default(); len];
        std::ptr::copy_nonoverlapping(
            data.cast::<u8>(),
            vec.as_mut_ptr().cast::<u8>(),
            len * std::mem::size_of::<T>(),
        );
        let array = ndarray::ArrayD::from_shape_vec(shape, vec)
            .expect("invalid shape")
            .into_shared();
        Box::new(array)
    }
}

struct MmapNdarray {
    mmap_ptr: *mut c_void,
    inner: Box<dyn Array>,
}

impl MmapNdarray {
    unsafe fn new(
        mmap_ptr: *mut c_void,
        data: *const c_void,
        shape: Vec<usize>,
        dtype: dlpk::sys::DLDataType,
    ) -> Self {
        use dlpk::sys::DLDataTypeCode;
        let inner: Box<dyn Array> = match (dtype.code, dtype.bits) {
            (DLDataTypeCode::kDLFloat, 32) => mmap_array_typed::<f32>(data, shape),
            (DLDataTypeCode::kDLFloat, 64) => mmap_array_typed::<f64>(data, shape),
            (DLDataTypeCode::kDLInt, 8) => mmap_array_typed::<i8>(data, shape),
            (DLDataTypeCode::kDLInt, 16) => mmap_array_typed::<i16>(data, shape),
            (DLDataTypeCode::kDLInt, 32) => mmap_array_typed::<i32>(data, shape),
            (DLDataTypeCode::kDLInt, 64) => mmap_array_typed::<i64>(data, shape),
            (DLDataTypeCode::kDLUInt, 8) => mmap_array_typed::<u8>(data, shape),
            (DLDataTypeCode::kDLUInt, 16) => mmap_array_typed::<u16>(data, shape),
            (DLDataTypeCode::kDLUInt, 32) => mmap_array_typed::<u32>(data, shape),
            (DLDataTypeCode::kDLUInt, 64) => mmap_array_typed::<u64>(data, shape),
            _ => panic!(
                "unsupported dtype for mmap: code={:?} bits={}",
                dtype.code, dtype.bits
            ),
        };
        Self { mmap_ptr, inner }
    }
}

impl Drop for MmapNdarray {
    fn drop(&mut self) {
        unsafe {
            crate::c_api::mts_mmap_free(self.mmap_ptr);
        }
    }
}

unsafe impl Send for MmapNdarray {}
unsafe impl Sync for MmapNdarray {}

impl Array for MmapNdarray {
    // Delegate as_any to the inner typed ArcArray so move_samples_from
    // can downcast to the correct concrete type.
    fn as_any(&self) -> &dyn std::any::Any { self.inner.as_any() }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self.inner.as_any_mut() }

    fn create(&self, shape: &[usize]) -> Box<dyn Array> { self.inner.create(shape) }
    fn copy(&self) -> Box<dyn Array> { self.inner.copy() }
    fn shape(&self) -> &[usize] { self.inner.shape() }

    fn reshape(&mut self, shape: &[usize]) { self.inner.reshape(shape) }
    fn swap_axes(&mut self, axis_1: usize, axis_2: usize) { self.inner.swap_axes(axis_1, axis_2) }

    fn move_samples_from(
        &mut self,
        input: &dyn Array,
        samples: &[crate::c_api::mts_sample_mapping_t],
        property: std::ops::Range<usize>,
    ) {
        self.inner.move_samples_from(input, samples, property);
    }

    fn device(&self) -> dlpk::sys::DLDevice { self.inner.device() }
    fn dtype(&self) -> dlpk::sys::DLDataType { self.inner.dtype() }

    fn as_dlpack(
        &self,
        device: dlpk::sys::DLDevice,
        stream: Option<i64>,
        max_version: dlpk::sys::DLPackVersion,
    ) -> Result<dlpk::DLPackTensor, Error> {
        self.inner.as_dlpack(device, stream, max_version)
    }
}
