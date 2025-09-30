use std::ops::Range;
use std::os::raw::c_void;
use std::sync::Mutex;

use once_cell::sync::Lazy;
use dlpack::sys::{DLManagedTensorVersioned, DLDataTypeCode};

use crate::c_api::mts_status_t;
use crate::Error;

/// A single 64-bit integer representing a data origin (numpy ndarray, rust
/// ndarray, torch tensor, fortran array, ...).
#[repr(transparent)]
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct mts_data_origin_t(pub u64);

static REGISTERED_DATA_ORIGIN: Lazy<Mutex<Vec<String>>> = Lazy::new(|| {
    // start the registered origins at 1, this allow using 0 as a marker for
    // "unknown data origin"
    Mutex::new(vec!["unregistered origin".into()])
});

/// Register a new data origin with the given `name`, or get the
/// `DataOrigin` corresponding to this name if it was already registered.
pub fn register_data_origin(name: String) -> mts_data_origin_t {
    let mut registered_origins = REGISTERED_DATA_ORIGIN.lock().expect("mutex got poisoned");

    for (i, registered) in registered_origins.iter().enumerate() {
        if registered == &name {
            return mts_data_origin_t(i as u64);
        }
    }

    // could not find the origin, register a new one
    registered_origins.push(name);

    return mts_data_origin_t((registered_origins.len() - 1) as u64);
}

/// Get the name of the given (pre-registered) origin
#[allow(clippy::cast_possible_truncation)]
pub fn get_data_origin(origin: mts_data_origin_t) -> String {
    let registered_origins = REGISTERED_DATA_ORIGIN.lock().expect("mutex got poisoned");
    let id = origin.0 as usize;

    if id < registered_origins.len() {
        return registered_origins[id].clone();
    } else {
        return registered_origins[0].clone();
    }
}

// SAFETY: this should be checked by the user/implementer of `mts_array_t`.
unsafe impl Sync for mts_array_t {}
unsafe impl Send for mts_array_t {}

/// `mts_array_t` manages n-dimensional arrays used as data in a block or tensor
/// map. The array itself is opaque to this library and can come from multiple
/// sources: Rust program, a C/C++ program, a Fortran program, Python with numpy
/// or torch. The data does not have to live on CPU, or even on the same machine
/// where this code is executed.
///
/// This struct contains a C-compatible manual implementation of a virtual table
/// (vtable, i.e. trait in Rust, pure virtual class in C++); allowing
/// manipulation of the array in an opaque way.
///
/// **WARNING**: all function implementations **MUST** be thread-safe, and can
/// be called from multiple threads at the same time. The `mts_array_t` itself
/// might be moved from one thread to another.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct mts_array_t {
    /// User-provided data should be stored here, it will be passed as the
    /// first parameter to all function pointers below.
    pub ptr: *mut c_void,

    /// This function needs to store the "data origin" for this array in
    /// `origin`. Users of `mts_array_t` should register a single data
    /// origin with `mts_register_data_origin`, and use it for all compatible
    /// arrays.
    origin: Option<unsafe extern "C" fn(
        array: *const c_void,
        origin: *mut mts_data_origin_t
    ) -> mts_status_t>,

    /// Get a pointer to the underlying data storage.
    ///
    /// This function is allowed to fail if the data is not accessible in RAM,
    /// not stored as 64-bit floating point values, or not stored as a
    /// C-contiguous array.
    data: Option<unsafe extern "C" fn(
        array: *mut c_void,
        data: *mut *mut f64,
    ) -> mts_status_t>,

    /// Get the shape of the array managed by this `mts_array_t` in the `*shape`
    /// pointer, and the number of dimension (size of the `*shape` array) in
    /// `*shape_count`.
    shape: Option<unsafe extern "C" fn(
        array: *const c_void,
        shape: *mut *const usize,
        shape_count: *mut usize,
    ) -> mts_status_t>,

    /// Change the shape of the array managed by this `mts_array_t` to the given
    /// `shape`. `shape_count` must contain the number of elements in the
    /// `shape` array
    reshape: Option<unsafe extern "C" fn(
        array: *mut c_void,
        shape: *const usize,
        shape_count: usize,
    ) -> mts_status_t>,

    /// Swap the axes `axis_1` and `axis_2` in this `array`.
    swap_axes: Option<unsafe extern "C" fn(
        array: *mut c_void,
        axis_1: usize,
        axis_2: usize,
    ) -> mts_status_t>,

    /// Create a new array with the same options as the current one (data type,
    /// data location, etc.) and the requested `shape`; and store it in
    /// `new_array`. The number of elements in the `shape` array should be given
    /// in `shape_count`.
    ///
    /// The new array should be filled with zeros.
    create: Option<unsafe extern "C" fn(
        array: *const c_void,
        shape: *const usize,
        shape_count: usize,
        new_array: *mut mts_array_t,
    ) -> mts_status_t>,

    /// Make a copy of this `array` and return the new array in `new_array`.
    ///
    /// The new array is expected to have the same data origin and parameters
    /// (data type, data location, etc.)
    copy: Option<unsafe extern "C" fn(
        array: *const c_void,
        new_array: *mut mts_array_t,
    ) -> mts_status_t>,

    /// Remove this array and free the associated memory. This function can be
    /// set to `NULL` is there is no memory management to do.
    destroy: Option<unsafe extern "C" fn(array: *mut c_void)>,

    /// Set entries in the `output` array (the current array) taking data from
    /// the `input` array. The `output` array is guaranteed to be created by
    /// calling `mts_array_t::create` with one of the arrays in the same block
    /// or tensor map as the `input`.
    ///
    /// The `samples` array of size `samples_count` indicate where the data
    /// should be moved from `input` to `output`.
    ///
    /// This function should copy data from `input[samples[i].input, ..., :]` to
    /// `array[samples[i].output, ..., property_start:property_end]` for `i` up
    /// to `samples_count`. All indexes are 0-based.
    move_samples_from: Option<unsafe extern "C" fn(
        output: *mut c_void,
        input: *const c_void,
        samples: *const mts_sample_mapping_t,
        samples_count: usize,
        property_start: usize,
        property_end: usize,
    ) -> mts_status_t>,
}

// DLPack internal form
struct MtsArrayInternal {
    tensor: *mut DLManagedTensorVersioned,
    origin_id: mts_data_origin_t,
}

impl mts_array_t {
    pub fn from_dlpack(dl_tensor: DLManagedTensorVersioned, origin_name: &str) -> mts_array_t {
        let boxed_tensor = Box::new(dl_tensor);
        let internal_state = Box::new(MtsArrayInternal {
            tensor: Box::into_raw(boxed_tensor),
            origin_id: register_data_origin(origin_name.into()),
        });
        mts_array_t {
            ptr: Box::into_raw(internal_state) as *mut c_void,
            origin: Some(shims::origin),
            data: Some(shims::data),
            shape: Some(shims::shape),
            reshape: Some(shims::reshape_unsupported),
            swap_axes: Some(shims::swap_axes_unsupported),
            create: Some(shims::create_unsupported),
            copy: Some(shims::copy_unsupported),
            destroy: Some(shims::destroy),
            move_samples_from: None,
        }
    }
}


mod shims {
    use super::*;
    use crate::c_api::{MTS_SUCCESS,MTS_INVALID_PARAMETER_ERROR,MTS_NOT_IMPLEMENTED_ERROR};

    fn log_deprecation() {
        eprintln!("Warning: This array operation is deprecated and will be removed.");
    }

    pub(super) unsafe extern "C" fn origin(array: *const c_void, origin: *mut mts_data_origin_t) -> mts_status_t {
        let internal = &*(array as *const MtsArrayInternal);
        *origin = internal.origin_id;
        mts_status_t(MTS_SUCCESS)
    }

    pub(super) unsafe extern "C" fn data(array: *mut c_void, data: *mut *mut f64) -> mts_status_t {
        let internal = &*(array as *const MtsArrayInternal);
        let dl = &(*internal.tensor).dl_tensor;
        if dl.dtype.code != DLDataTypeCode::kDLFloat || dl.dtype.bits != 64 {
            return mts_status_t(MTS_INVALID_PARAMETER_ERROR);
        }
        *data = (dl.data as *mut u8).add(dl.byte_offset as usize) as *mut f64;
        mts_status_t(MTS_SUCCESS)
    }

    pub(super) unsafe extern "C" fn shape(array: *const c_void, shape: *mut *const usize, count: *mut usize) -> mts_status_t {
        let internal = &*(array as *const MtsArrayInternal);
        let dl = &(*internal.tensor).dl_tensor;
        *shape = dl.shape as *const usize;
        *count = dl.ndim as usize;
        mts_status_t(MTS_SUCCESS)
    }

    pub(super) unsafe extern "C" fn destroy(array: *mut c_void) {
        if array.is_null() { return; }
        let internal_boxed = Box::from_raw(array as *mut MtsArrayInternal);
        let tensor_ptr = internal_boxed.tensor;
        if let Some(deleter) = (*tensor_ptr).deleter {
            deleter(tensor_ptr);
        } else {
            let _ = Box::from_raw(tensor_ptr);
        }
    }

    pub(super) unsafe extern "C" fn reshape_unsupported(_: *mut c_void, _: *const usize, _: usize) -> mts_status_t {
        log_deprecation(); mts_status_t(MTS_NOT_IMPLEMENTED_ERROR)
    }
    pub(super) unsafe extern "C" fn swap_axes_unsupported(_: *mut c_void, _: usize, _: usize) -> mts_status_t {
        log_deprecation(); mts_status_t(MTS_NOT_IMPLEMENTED_ERROR)
    }
    pub(super) unsafe extern "C" fn create_unsupported(_: *const c_void, _: *const usize, _: usize, _: *mut mts_array_t) -> mts_status_t {
        log_deprecation(); mts_status_t(MTS_NOT_IMPLEMENTED_ERROR)
    }
    pub(super) unsafe extern "C" fn copy_unsupported(_: *const c_void, _: *mut mts_array_t) -> mts_status_t {
        log_deprecation(); mts_status_t(MTS_NOT_IMPLEMENTED_ERROR)
    }
}

/// Representation of a single sample moved from an array to another one
#[derive(Debug, Clone)]
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct mts_sample_mapping_t {
    /// index of the moved sample in the input array
    pub input: usize,
    /// index of the moved sample in the output array
    pub output: usize,
}

impl std::fmt::Debug for mts_array_t {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut origin = None;
        let mut shape = None;
        if !self.ptr.is_null() {
            origin = self.origin().ok().map(get_data_origin);
            shape = self.shape().ok();
        }

        f.debug_struct("mts_array_t")
            .field("ptr", &self.ptr)
            .field("origin", &origin)
            .field("shape", &shape)
            .finish_non_exhaustive()
    }
}

impl Drop for mts_array_t {
    fn drop(&mut self) {
        if let Some(function) = self.destroy {
            unsafe { function(self.ptr) }
        }
    }
}

impl mts_array_t {
    /// make a raw (member by member) copy of the array. Contrary to
    /// `mts_array_t::clone`, the returned array refers to the same
    /// `mts_array_t` instance, and as such should not be freed.
    pub(crate) fn raw_copy(&self) -> mts_array_t {
        mts_array_t {
            ptr: self.ptr,
            origin: self.origin,
            data: self.data,
            shape: self.shape,
            reshape: self.reshape,
            swap_axes: self.swap_axes,
            create: self.create,
            copy: self.copy,
            // do not copy destroy, the user should never call it
            destroy: None,
            move_samples_from: self.move_samples_from,
        }
    }

    /// Create an `mts_array_t` with all fields set to null pointers.
    pub(crate) fn null() -> mts_array_t {
        mts_array_t {
            ptr: std::ptr::null_mut(),
            origin: None,
            data: None,
            shape: None,
            reshape: None,
            swap_axes: None,
            create: None,
            copy: None,
            destroy: None,
            move_samples_from: None,
        }
    }

    /// Get the origin of this array
    pub fn origin(&self) -> Result<mts_data_origin_t, Error> {
        let function = self.origin.expect("mts_array_t.origin function is NULL");

        let mut origin = mts_data_origin_t(0);
        let status = unsafe {
            function(self.ptr, &mut origin)
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling mts_array_t.origin failed".into()
            });
        }

        return Ok(origin);
    }

    /// Get the underlying data for this array.
    pub fn data(&self) -> Result<&[f64], Error> {
        let shape = self.shape()?;
        let mut len = 1;
        for s in shape {
            len *= s;
        }

        let function = self.data.expect("mts_array_t.data function is NULL");

        let mut data_ptr = std::ptr::null_mut();

        let status = unsafe {
            function(
                self.ptr,
                &mut data_ptr,
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling mts_array_t.data failed".into()
            });
        }

        if len == 0 {
            let data: &[f64] = &[];
            return Ok(data);
        }

        assert!(!data_ptr.is_null());
        let data = unsafe {
            std::slice::from_raw_parts(data_ptr, len)
        };

        return Ok(data);
    }

    /// Get the underlying data for this array.
    pub fn data_mut(&mut self) -> Result<&mut [f64], Error> {
        let shape = self.shape()?;
        let mut len = 1;
        for s in shape {
            len *= s;
        }

        let function = self.data.expect("mts_array_t.data function is NULL");

        let mut data_ptr = std::ptr::null_mut();

        let status = unsafe {
            function(
                self.ptr,
                &mut data_ptr,
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling mts_array_t.data failed".into()
            });
        }

        if len == 0 {
            let data: &mut [f64] = &mut [];
            return Ok(data);
        }

        assert!(!data_ptr.is_null());
        let data = unsafe {
            std::slice::from_raw_parts_mut(data_ptr, len)
        };

        return Ok(data);
    }

    /// Get the shape of this array
    #[allow(clippy::cast_possible_truncation)]
    pub fn shape(&self) -> Result<&[usize], Error> {
        let function = self.shape.expect("mts_array_t.shape function is NULL");

        let mut shape = std::ptr::null();
        let mut shape_count: usize = 0;

        let status = unsafe {
            function(
                self.ptr,
                &mut shape,
                &mut shape_count,
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling mts_array_t.shape failed".into()
            });
        }

        assert!(shape_count > 0);
        assert!(!shape.is_null());
        let shape = unsafe {
            std::slice::from_raw_parts(shape, shape_count)
        };

        return Ok(shape);
    }

    /// Set the shape of this array to the given new `shape`
    pub fn reshape(&mut self, shape: &[usize]) -> Result<(), Error> {
        let function = self.reshape.expect("mts_array_t.reshape function is NULL");

        let status = unsafe {
            function(
                self.ptr,
                shape.as_ptr(),
                shape.len(),
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling mts_array_t.reshape failed".into()
            });
        }

        return Ok(());
    }

    /// Swap the axes `axis_1` and `axis_2` in the dimensions of this array.
    pub fn swap_axes(&mut self, axis_1: usize, axis_2: usize) -> Result<(), Error> {
        let function = self.swap_axes.expect("mts_array_t.swap_axes function is NULL");

        let status = unsafe {
            function(
                self.ptr,
                axis_1,
                axis_2,
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling mts_array_t.swap_axes failed".into()
            });
        }

        return Ok(());
    }

    /// Create a new array with the same settings as this one and the given `shape`
    pub fn create(&self, shape: &[usize]) -> Result<mts_array_t, Error> {
        let function = self.create.expect("mts_array_t.create function is NULL");

        let mut data_storage = mts_array_t::null();
        let status = unsafe {
            function(
                self.ptr,
                shape.as_ptr(),
                shape.len(),
                &mut data_storage
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling mts_array_t.create failed".into()
            });
        }

        return Ok(data_storage);
    }

    /// Try to copy this `mts_array_t`. This can fail if the external data can
    /// not be copied for some reason
    pub fn try_clone(&self) -> Result<mts_array_t, Error> {
        let function = self.copy.expect("mts_array_t.copy function is NULL");

        let mut new_array = mts_array_t::null();
        let status = unsafe {
            function(self.ptr, &mut new_array)
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling mts_array_t.create failed".into()
            });
        }

        return Ok(new_array);
    }

    /// Set entries in `self` (the current array) taking data from the `input`
    /// array. The `self` array is guaranteed to be created by calling
    /// `Array::create` with one of the arrays in the same block or tensor
    /// map as the `input`.
    ///
    /// The `samples` array indicate where the data should be moved from `input`
    /// to `output`.
    ///
    /// This function should copy data from `input[sample.input, ..., :]` to
    /// `array[sample.output, ..., properties]` for all `sample` in `samples`.
    /// All indexes are 0-based.
    pub fn move_samples_from(
        &mut self,
        input: &mts_array_t,
        samples: &[mts_sample_mapping_t],
        properties: Range<usize>,
    ) -> Result<(), Error> {
        let function = self.move_samples_from.expect("mts_array_t.move_samples_from function is NULL");

        let status = unsafe {
            function(
                self.ptr,
                input.ptr,
                samples.as_ptr(),
                samples.len(),
                properties.start,
                properties.end,
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling mts_array_t.move_samples_from failed".into()
            });
        }

        return Ok(());
    }
}

#[cfg(test)]
pub(crate) use self::tests::TestArray;

#[cfg(test)]
mod tests {
    use crate::c_api::MTS_NOT_IMPLEMENTED_ERROR;
    use super::*;
    use dlpack::sys as dl;

    fn mock_dlpack_tensor(shape: Vec<i64>, data: Vec<f64>) -> DLManagedTensorVersioned {
        let data_ptr = Box::leak(data.into_boxed_slice()).as_mut_ptr();
        let ndim = shape.len();
        let shape_ptr = Box::leak(shape.into_boxed_slice()).as_mut_ptr();

        unsafe extern "C" fn mock_deleter(managed: *mut DLManagedTensorVersioned) {
            if managed.is_null() { return; }
            let tensor = &(*managed).dl_tensor;

            let shape_slice = std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize);
            let data_len = shape_slice.iter().product::<i64>() as usize;
            let data_ptr = std::ptr::slice_from_raw_parts_mut(tensor.data as *mut f64, data_len);
            let _ = Box::from_raw(data_ptr);

            let shape_ptr = std::ptr::slice_from_raw_parts_mut(tensor.shape, tensor.ndim as usize);
            let _ = Box::from_raw(shape_ptr);

            let _ = Box::from_raw(managed);
        }

        DLManagedTensorVersioned {
            version: dl::DLPackVersion { major: 1, minor: 0 },
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(mock_deleter),
            flags: 0,
            dl_tensor: dl::DLTensor {
                data: data_ptr as *mut c_void,
                device: dl::DLDevice { device_type: dl::DLDeviceType::kDLCPU, device_id: 0 },
                ndim: ndim as i32,
                dtype: dl::DLDataType { code: DLDataTypeCode::kDLFloat, bits: 64, lanes: 1 },
                shape: shape_ptr,
                strides: std::ptr::null_mut(),
                byte_offset: 0,
            },
        }
    }

    pub struct TestArray;

    impl TestArray {
        #[allow(clippy::new_ret_no_self)]
        pub(crate) fn new(shape: Vec<usize>) -> mts_array_t {
            let shape_i64 = shape.iter().map(|&s| s as i64).collect();
            let data = vec![0.0; shape.iter().product()];
            let dl_tensor = mock_dlpack_tensor(shape_i64, data);
            mts_array_t::from_dlpack(dl_tensor, "rust.TestArray")
        }

        #[allow(clippy::new_ret_no_self)]
        pub(crate) fn new_other_origin(shape: Vec<usize>) -> mts_array_t {
            let shape_i64 = shape.iter().map(|&s| s as i64).collect();
            let data = vec![0.0; shape.iter().product()];
            let dl_tensor = mock_dlpack_tensor(shape_i64, data);
            mts_array_t::from_dlpack(dl_tensor, "rust.TestArrayOtherOrigin")
        }
    }

    #[test]
    fn facade_works() {
        let shape = vec![2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dl = mock_dlpack_tensor(shape.iter().map(|&s| s as i64).collect(), data.clone());
        let array = mts_array_t::from_dlpack(dl, "rust.TestArray");

        let origin_id = array.origin().unwrap();
        assert_eq!(get_data_origin(origin_id), "rust.TestArray");
        assert_eq!(array.shape().unwrap(), &[2, 3]);
        assert_eq!(array.data().unwrap(), &data);

        let mut mut_array = array.raw_copy();
        let err = mut_array.reshape(&[6]).unwrap_err();
        assert!(matches!(err, Error::External { status, .. } if status.0 == MTS_NOT_IMPLEMENTED_ERROR));
    }

    #[test]
    fn data_origin() {
        assert_eq!(get_data_origin(mts_data_origin_t(0)), "unregistered origin");
        assert_eq!(get_data_origin(mts_data_origin_t(10000)), "unregistered origin");

        let origin = register_data_origin("test origin".into());
        assert_eq!(get_data_origin(origin), "test origin");
    }

    #[test]
    fn debug() {
        let data: mts_array_t = TestArray::new(vec![3, 4, 5]);

        let debug_format = format!("{:?}", data);
        assert_eq!(debug_format, format!(
            "mts_array_t {{ ptr: {:?}, origin: Some(\"rust.TestArray\"), shape: Some([3, 4, 5]), .. }}",
            data.ptr
        ));
    }
}
