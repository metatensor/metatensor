use std::ffi::c_void;
use dlpack::GetDLPackDataType;
use dlpack::sys as dl;
use num_traits::NumCast;

use crate::mts_array_t;

pub struct TestArray;

impl TestArray {
    #[allow(clippy::new_ret_no_self)]
    pub fn new<T>(shape: Vec<usize>) -> mts_array_t
    where
        T: Clone + Default + GetDLPackDataType + NumCast
    {
        let mut data = vec![T::default(); shape.iter().product()];
        for i in 0..data.len() {
            data[i] = T::from(i as i32 + 1).unwrap();
        }

        let shape_i64 = shape.iter().map(|&s| s as i64).collect();
        let dl_tensor = mock_typed_dlpack_tensor(shape_i64, data);
        mts_array_t::from_dlpack(dl_tensor, "rust.TestArray")
    }

    #[allow(clippy::new_ret_no_self)]
    pub(crate) fn new_other_origin(shape: Vec<usize>) -> mts_array_t {
        let shape_i64 = shape.iter().map(|&s| s as i64).collect();
        let data = vec![0.0; shape.iter().product()];
        let dl_tensor = mock_typed_dlpack_tensor(shape_i64, data);
        mts_array_t::from_dlpack(dl_tensor, "rust.TestArrayOtherOrigin")
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn new_typed<T>(shape: Vec<usize>, origin_name: &str) -> (mts_array_t, Vec<T>)
    where
        T: Clone + Default + GetDLPackDataType + NumCast
    {
        let mut data = vec![T::default(); shape.iter().product()];
        for i in 0..data.len() {
            data[i] = T::from(i as i32 + 1).unwrap();
        }

        let shape_i64 = shape.iter().map(|&s| s as i64).collect();
        let dl_tensor = mock_typed_dlpack_tensor(shape_i64, data.clone());
        let array = mts_array_t::from_dlpack(dl_tensor, origin_name);

        (array, data)
    }
}


fn mock_typed_dlpack_tensor<T: Clone + GetDLPackDataType>(shape: Vec<i64>, data: Vec<T>) -> dl::DLManagedTensorVersioned {
    let data_ptr = Box::leak(data.into_boxed_slice()).as_mut_ptr();
    let ndim = shape.len();
    let shape_ptr = Box::leak(shape.into_boxed_slice()).as_mut_ptr();

    unsafe extern "C" fn mock_deleter<T>(managed: *mut dl::DLManagedTensorVersioned) {
        if managed.is_null() { return; }
        let tensor = &(*managed).dl_tensor;

        let shape_slice = std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize);
        let data_len = shape_slice.iter().product::<i64>() as usize;

        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(tensor.data as *mut T, data_len));
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(tensor.shape, tensor.ndim as usize));
        let _ = Box::from_raw(managed);
    }

    let deleter = mock_deleter::<T>;

    dl::DLManagedTensorVersioned {
        version: dl::DLPackVersion { major: 1, minor: 0 },
        manager_ctx: std::ptr::null_mut(),
        deleter: Some(deleter),
        flags: 0,
        dl_tensor: dl::DLTensor {
            data: data_ptr as *mut c_void,
            device: dl::DLDevice { device_type: dl::DLDeviceType::kDLCPU, device_id: 0 },
            ndim: ndim as i32,
            dtype: T::get_dlpack_data_type(),
            shape: shape_ptr,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
    }
}
