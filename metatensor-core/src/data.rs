use std::os::raw::c_void;
use std::sync::Mutex;
use std::ptr::NonNull;

use once_cell::sync::Lazy;

use dlpk::sys::{DLDataType, DLDevice, DLManagedTensorVersioned, DLPackVersion};
use dlpk::DLPackTensor;

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

    /// Remove this array and free the associated memory. This function can be
    /// set to `NULL` if there is no memory management to do.
    pub(crate) destroy: Option<unsafe extern "C" fn(array: *mut c_void)>,

    /// This function needs to store the "data origin" for this array in
    /// `origin`. Users of `mts_array_t` should register a single data
    /// origin with `mts_register_data_origin`, and use it for all compatible
    /// arrays.
    pub(crate) origin: Option<unsafe extern "C" fn(
        array: *const c_void,
        origin: *mut mts_data_origin_t
    ) -> mts_status_t>,

    /// Query the device where this array's data resides without exporting
    /// via DLPack.
    ///
    /// The implementation must store the device information in `*device`.
    pub(crate) device: Option<unsafe extern "C" fn(
        array: *const c_void,
        device: *mut DLDevice,
    ) -> mts_status_t>,

    /// Query the data type of this array without a full DLPack export.
    ///
    /// The implementation must store the data type in `*dtype`.
    pub(crate) dtype: Option<unsafe extern "C" fn(
        array: *const c_void,
        dtype: *mut DLDataType,
    ) -> mts_status_t>,

    /// Get a DLPack representation of the underlying data.
    ///
    /// This function exports the array as a `DLManagedTensorVersioned` struct
    /// into `*dl_managed_tensor`, following the DLPack data interchange
    /// standard.
    ///
    /// The `device` parameter specifies the desired DLPack device type. If this
    /// differs from the array's current device, the implementation should
    /// attempt to make the data accessible on the requested device (e.g., by
    /// copying).
    ///
    /// The `stream` parameter is a pointer to an integer representing a
    /// device-specific stream or queue. If this is `NULL`, the implementation
    /// should use the default stream for the specified device. If this is `-1`,
    /// no synchronization should be performed. Some devices have specific
    /// stream values:
    /// - For CUDA devices, `1` represents the legacy default stream, `2` the
    ///   per-thread default stream. Any value above `2` indicates the stream
    ///   number. `0` is not allowed as it could mean the same as `NULL`, `1` or
    ///   `2`.
    /// - For ROCm devices, `0` represents the default stream, any value above
    ///   `2` indicates the stream number. `1` and `2` are not allowed.
    ///
    /// See also the documentation of `__dlpack__` for more information about
    /// streams:
    /// <https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html>
    ///
    /// `max_version` specifies the maximum DLPack API version the caller
    /// supports. The implementation should try to return a tensor compatible
    /// with this version, but this is not guaranteed, and the caller should
    /// check the returned tensor's version.
    ///
    /// The returned `DLManagedTensorVersioned` is owned by the caller, who is
    /// responsible for calling its `deleter` function when the tensor is no
    /// longer needed. The lifetime of the `DLManagedTensorVersioned` must not
    /// exceed the lifetime of the `mts_array_t` it was created from.
    pub(crate) as_dlpack: Option<unsafe extern "C" fn(
        array: *mut c_void,
        dl_managed_tensor: *mut *mut DLManagedTensorVersioned,
        device: DLDevice,
        stream: *const i64,
        max_version: DLPackVersion,
    ) -> mts_status_t>,

    /// Get the shape of the array managed by this `mts_array_t` in the `*shape`
    /// pointer, and the number of dimension (size of the `*shape` array) in
    /// `*shape_count`.
    pub(crate) shape: Option<unsafe extern "C" fn(
        array: *const c_void,
        shape: *mut *const usize,
        shape_count: *mut usize,
    ) -> mts_status_t>,

    /// Change the shape of the array managed by this `mts_array_t` to the given
    /// `shape`. `shape_count` must contain the number of elements in the
    /// `shape` array
    pub(crate) reshape: Option<unsafe extern "C" fn(
        array: *mut c_void,
        shape: *const usize,
        shape_count: usize,
    ) -> mts_status_t>,

    /// Swap the axes `axis_1` and `axis_2` in this `array`.
    pub(crate) swap_axes: Option<unsafe extern "C" fn(
        array: *mut c_void,
        axis_1: usize,
        axis_2: usize,
    ) -> mts_status_t>,

    /// Create a new array with the same options as the current one (data type,
    /// data location, etc.) and the requested `shape`; and store it in
    /// `new_array`. The number of elements in the `shape` array should be given
    /// in `shape_count`.
    ///
    /// The new array should be filled with the scalar value from `fill_value`,
    /// which must be an `mts_array_t` with shape `(1,)` and the same dtype as
    /// this array. This function should call `fill_value.destroy` if the
    /// function pointer is not null when `fill_value` is no longer needed.
    pub(crate) create: Option<unsafe extern "C" fn(
        array: *const c_void,
        shape: *const usize,
        shape_count: usize,
        fill_value: mts_array_t,
        new_array: *mut mts_array_t,
    ) -> mts_status_t>,

    /// Make a copy of this `array` and return the new array in `new_array`.
    ///
    /// The new array is expected to have the same data origin and parameters
    /// (data type, data location, etc.)
    pub(crate) copy: Option<unsafe extern "C" fn(
        array: *const c_void,
        new_array: *mut mts_array_t,
    ) -> mts_status_t>,

    /// Set entries in the `output` array (the current array) taking data from
    /// the `input` array. The `output` array is guaranteed to be created by
    /// calling `mts_array_t::create` with one of the arrays in the same block
    /// or tensor map as the `input`.
    ///
    /// The `movements` array of size `movements_count` indicate where the data
    /// should be moved from `input` to `output`.
    ///
    /// This function should copy data from `input[movements[i].sample_in, ...,
    /// movements[i].properties_start_in + x]` to
    /// `array[movements[i].sample_out, ..., movements[i].properties_start_out +
    /// x]` for `i` up to `movements_count` and `x` up to
    /// `movements[i].properties_length`. All indexes are 0-based.
    pub(crate) move_data: Option<unsafe extern "C" fn(
        output: *mut c_void,
        input: *const c_void,
        movements: *const mts_data_movement_t,
        movements_count: usize,
    ) -> mts_status_t>,
}

/// Information about a block of data to be moved from an array to another.
#[derive(Debug, Clone)]
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct mts_data_movement_t {
    /// index of the sample in the input array
    pub sample_in: usize,
    /// index of the sample in the output array
    pub sample_out: usize,
    /// index of the start of the property in the input array
    pub properties_start_in: usize,
    /// index of the start of the property in the output array
    pub properties_start_out: usize,
    /// number of properties to move
    pub properties_length: usize,
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
            device: self.device,
            dtype: self.dtype,
            as_dlpack: self.as_dlpack,
            shape: self.shape,
            reshape: self.reshape,
            swap_axes: self.swap_axes,
            create: self.create,
            copy: self.copy,
            // do not copy destroy, the user should never call it
            destroy: None,
            move_data: self.move_data,
        }
    }

    /// Create an `mts_array_t` with all fields set to null pointers.
    pub(crate) fn null() -> mts_array_t {
        mts_array_t {
            ptr: std::ptr::null_mut(),
            origin: None,
            device: None,
            dtype: None,
            as_dlpack: None,
            shape: None,
            reshape: None,
            swap_axes: None,
            create: None,
            copy: None,
            destroy: None,
            move_data: None,
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

    /// Get the device where this array's data resides
    pub fn device(&self) -> Result<DLDevice, Error> {
        let function = self.device.expect("mts_array_t.device function is NULL");

        let mut device = DLDevice::cpu();
        let status = unsafe {
            function(self.ptr, &mut device)
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling mts_array_t.device failed".into()
            });
        }

        return Ok(device);
    }

    /// Get the data type of this array.
    pub fn dtype(&self) -> Result<DLDataType, Error> {
        let function = self.dtype.expect("mts_array_t.dtype function is NULL");

        let mut dtype = DLDataType { code: dlpk::sys::DLDataTypeCode::kDLFloat, bits: 64, lanes: 1 };
        let status = unsafe { function(self.ptr, &mut dtype) };
        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling mts_array_t.dtype failed".into()
            });
        }
        return Ok(dtype);
    }

    /// Get a dlpack representation of the data
    pub fn as_dlpack(
        &self,
        device: DLDevice,
        stream: Option<i64>,
        max_version: DLPackVersion
    ) -> Result<DLPackTensor, Error> {
        // C function pointer from the vtable slot
        let function = self.as_dlpack.expect("mts_array_t.as_dlpack function is NULL");
        // ... and fill structure
        let mut dl_managed_tensor: *mut DLManagedTensorVersioned = std::ptr::null_mut();

        let stream_c = match stream {
            Some(s) => &s as *const i64,
            None => std::ptr::null(),
        };
        let status = unsafe {
            function(self.ptr, &mut dl_managed_tensor, device, stream_c, max_version)
        };
        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling mts_array_t.as_dlpack failed".into()
            });
        }
        assert!(!dl_managed_tensor.is_null(), "mts_array_t.as_dlpack returned a null pointer on success");
        let ptr = NonNull::new(dl_managed_tensor).expect("pointer is null, this should not happen");
        let tensor = unsafe {
            DLPackTensor::from_ptr(ptr)
        };
        return Ok(tensor);
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

    /// Create a new array with the same settings as this one and the given
    /// `shape`, filled with the scalar from `fill_value`.
    ///
    /// `fill_value` must be a CPU `mts_array_t` with shape `(1,)` and the
    /// same dtype as this array.
    pub fn create(&self, shape: &[usize], fill_value: &mts_array_t) -> Result<mts_array_t, Error> {
        let self_dtype = self.dtype()?;
        let fill_dtype = fill_value.dtype()?;
        if self_dtype != fill_dtype {
            return Err(Error::InvalidParameter(format!(
                "fill_value dtype (code={}, bits={}, lanes={}) does not match \
                 array dtype (code={}, bits={}, lanes={})",
                fill_dtype.code as u32, fill_dtype.bits, fill_dtype.lanes,
                self_dtype.code as u32, self_dtype.bits, self_dtype.lanes,
            )));
        }

        let fill_shape = fill_value.shape()?;
        if fill_shape != [1] {
            return Err(Error::InvalidParameter(format!(
                "fill_value must have shape [1], got {:?}", fill_shape
            )));
        }

        let fill_device = fill_value.device()?;
        if fill_device.device_type != DLDevice::cpu().device_type {
            return Err(Error::InvalidParameter(
                "fill_value must be on CPU".into()
            ));
        }

        let function = self.create.expect("mts_array_t.create function is NULL");

        let mut data_storage = mts_array_t::null();
        let status = unsafe {
            function(
                self.ptr,
                shape.as_ptr(),
                shape.len(),
                // raw_copy() sets destroy to None, so the callback
                // receives a readable copy without taking ownership
                fill_value.raw_copy(),
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
    /// `Array::create` with one of the arrays in the same block or tensor map
    /// as the `input`.
    ///
    /// The `movements` array of size `movements_count` indicate where the data
    /// should be moved from `input` to `output`.
    ///
    /// This function should copy data from `input[movements[i].sample_in, ...,
    /// movements[i].properties_start_in + x]` to
    /// `array[movements[i].sample_out, ..., movements[i].properties_start_out +
    /// x]` for `i` up to `movements_count` and `x` up to
    /// `movements[i].properties_length`. All indexes are 0-based.
    pub fn move_data(
        &mut self,
        input: &mts_array_t,
        movements: &[mts_data_movement_t],
    ) -> Result<(), Error> {
        let function = self.move_data.expect("mts_array_t.move_data function is NULL");

        let status = unsafe {
            function(
                self.ptr,
                input.ptr,
                movements.as_ptr(),
                movements.len(),
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling mts_array_t.move_data failed".into()
            });
        }

        return Ok(());
    }
}

#[cfg(test)]
pub(crate) use self::tests::TestArray;

#[cfg(test)]
mod tests {
    use crate::c_api::MTS_SUCCESS;

    use super::*;

    pub struct TestArray {
        shape: Vec<usize>,
    }

    impl TestArray {
        #[allow(clippy::new_ret_no_self)]
        pub fn new(shape: Vec<usize>) -> mts_array_t {
            let array = Box::new(TestArray {shape});

            return mts_array_t {
                ptr: Box::into_raw(array).cast(),
                origin: Some(TestArray::origin),
                device: Some(TestArray::device_cpu),
                dtype: Some(TestArray::dtype_f64),
                as_dlpack: None,
                shape: Some(TestArray::shape),
                reshape: Some(TestArray::reshape),
                swap_axes: Some(TestArray::swap_axes),
                create: None,
                copy: None,
                destroy: Some(TestArray::destroy),
                move_data: None,
            }
        }

        pub fn new_other_origin(shape: Vec<usize>) -> mts_array_t {
            let array = Box::new(TestArray {shape});

            return mts_array_t {
                ptr: Box::into_raw(array).cast(),
                origin: Some(TestArray::other_origin),
                device: Some(TestArray::device_cpu),
                dtype: Some(TestArray::dtype_f64),
                as_dlpack: None,
                shape: Some(TestArray::shape),
                reshape: Some(TestArray::reshape),
                swap_axes: Some(TestArray::swap_axes),
                create: None,
                copy: None,
                destroy: Some(TestArray::destroy),
                move_data: None,
            }
        }

        /// Create a test array reporting a non-CPU device `(device_type=2, id=0)`
        pub fn new_other_device(shape: Vec<usize>) -> mts_array_t {
            let array = Box::new(TestArray {shape});

            return mts_array_t {
                ptr: Box::into_raw(array).cast(),
                origin: Some(TestArray::origin),
                device: Some(TestArray::device_cuda),
                dtype: Some(TestArray::dtype_f64),
                as_dlpack: None,
                shape: Some(TestArray::shape),
                reshape: Some(TestArray::reshape),
                swap_axes: Some(TestArray::swap_axes),
                create: None,
                copy: None,
                destroy: Some(TestArray::destroy),
                move_data: None,
            }
        }

        /// Create a test array reporting f32 dtype instead of f64
        pub fn new_other_dtype(shape: Vec<usize>) -> mts_array_t {
            let array = Box::new(TestArray {shape});

            return mts_array_t {
                ptr: Box::into_raw(array).cast(),
                origin: Some(TestArray::origin),
                device: Some(TestArray::device_cpu),
                dtype: Some(TestArray::dtype_f32),
                as_dlpack: None,
                shape: Some(TestArray::shape),
                reshape: Some(TestArray::reshape),
                swap_axes: Some(TestArray::swap_axes),
                create: None,
                copy: None,
                destroy: Some(TestArray::destroy),
                move_data: None,
            }
        }

        unsafe extern "C" fn origin(_: *const c_void, origin: *mut mts_data_origin_t) -> mts_status_t {
            *origin = register_data_origin("rust.TestArray".into());

            return mts_status_t(MTS_SUCCESS);
        }

        unsafe extern "C" fn other_origin(_: *const c_void, origin: *mut mts_data_origin_t) -> mts_status_t {
            *origin = register_data_origin("rust.TestArrayOtherOrigin".into());

            return mts_status_t(MTS_SUCCESS);
        }

        unsafe extern "C" fn device_cpu(_: *const c_void, device: *mut DLDevice) -> mts_status_t {
            *device = DLDevice::cpu();
            return mts_status_t(MTS_SUCCESS);
        }

        unsafe extern "C" fn device_cuda(_: *const c_void, device: *mut DLDevice) -> mts_status_t {
            *device = DLDevice { device_type: dlpk::sys::DLDeviceType::kDLCUDA, device_id: 0 };
            return mts_status_t(MTS_SUCCESS);
        }

        unsafe extern "C" fn dtype_f64(_: *const c_void, dtype: *mut DLDataType) -> mts_status_t {
            *dtype = DLDataType { code: dlpk::sys::DLDataTypeCode::kDLFloat, bits: 64, lanes: 1 };
            return mts_status_t(MTS_SUCCESS);
        }

        unsafe extern "C" fn dtype_f32(_: *const c_void, dtype: *mut DLDataType) -> mts_status_t {
            *dtype = DLDataType { code: dlpk::sys::DLDataTypeCode::kDLFloat, bits: 32, lanes: 1 };
            return mts_status_t(MTS_SUCCESS);
        }

        unsafe extern "C" fn shape(ptr: *const c_void, shape: *mut *const usize, shape_count: *mut usize) -> mts_status_t {
            let ptr = ptr.cast::<TestArray>();

            *shape = (*ptr).shape.as_ptr();
            *shape_count = (*ptr).shape.len();

            return mts_status_t(MTS_SUCCESS);
        }

        unsafe extern "C" fn reshape(ptr: *mut c_void, shape_ptr: *const usize, shape_count: usize) -> mts_status_t {
            let ptr = ptr.cast::<TestArray>();

            let shape = &mut (*ptr).shape;
            shape.clear();

            for i in 0..shape_count {
                shape.push(shape_ptr.add(i).read());
            }

            return mts_status_t(MTS_SUCCESS);
        }

        unsafe extern "C" fn swap_axes(ptr: *mut c_void, axis_1: usize, axis_2: usize) -> mts_status_t {
            let ptr = ptr.cast::<TestArray>();

            let shape = &mut (*ptr).shape;
            shape.swap(axis_1, axis_2);

            return mts_status_t(MTS_SUCCESS);
        }

        unsafe extern "C" fn destroy(ptr: *mut c_void) {
            let ptr = ptr.cast::<TestArray>();
            let boxed = Box::from_raw(ptr);
            std::mem::drop(boxed);
        }
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
