use std::os::raw::c_void;

use once_cell::sync::Lazy;

use dlpk::sys::{DLDevice, DLManagedTensorVersioned, DLPackVersion, DLDataType};
use dlpk::{DLPackTensor, GetDLPackDataType};

use crate::errors::Error;
use crate::c_api::{mts_array_t, mts_data_origin_t, mts_data_movement_t, mts_status_t};

/// The Array trait is used by metatensor to manage different kind of data array
/// with a single API. Metatensor only knows about `Box<dyn Array>`, and
/// manipulate the data through the functions on this trait.
///
/// This corresponds to the `mts_array_t` struct in metatensor-core.
pub trait Array: std::any::Any + Send + Sync {
    /// Get the array as a `Any` reference
    fn as_any(&self) -> &dyn std::any::Any;

    /// Get the array as a mutable `Any` reference
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Create a new array with the same options as the current one (data type,
    /// data location, etc.) and the requested `shape`.
    ///
    /// The new array should be filled with zeros.
    fn create(&self, shape: &[usize]) -> Box<dyn Array>;

    /// Make a copy of this `array`
    ///
    /// The new array is expected to have the same data origin and parameters
    /// (data type, data location, etc.)
    fn copy(&self) -> Box<dyn Array>;

    /// Get the shape of the array
    fn shape(&self) -> &[usize];

    /// Change the shape of the array to the given `shape`
    fn reshape(&mut self, shape: &[usize]);

    /// Swap the axes `axis_1` and `axis_2` in this array
    fn swap_axes(&mut self, axis_1: usize, axis_2: usize);

    /// Set entries in `self` taking data from the `input` array.
    ///
    /// The `output` array is guaranteed to be created by calling
    /// `mts_array_t::create` with one of the arrays in the same block or tensor
    /// map as the `input`.
    ///
    /// The `movements` indicate where the data should be moved from `input` to
    /// `output`.
    ///
    /// This function should copy data from `input[movements[i].sample_in, ...,
    /// movements[i].properties_start_in + x]` to
    /// `array[movements[i].sample_out, ..., movements[i].properties_start_out +
    /// x]` for `i` up to `movements_count` and `x` up to
    /// `movements[i].properties_length`. All indexes are 0-based.
    fn move_data(
        &mut self,
        input: &dyn Array,
        movements: &[mts_data_movement_t],
    );

    /// Get the device where this array's data resides.
    ///
    /// For CPU arrays this should return `DLDevice::cpu()`.
    fn device(&self) -> DLDevice;

    /// Get the data type of this array.
    ///
    /// This populates the `dtype` vtable slot for fast dtype queries.
    /// Implementations should return the appropriate `DLDataType` for their
    /// element type (e.g. float64 = `DLDataType { code: kDLFloat, bits: 64, lanes: 1 }`).
    fn dtype(&self) -> DLDataType;

    /// Convert the array to a `DLPack` tensor.
    /// The returned pointer is owned by the caller (and cleaned up via its deleter).
    fn as_dlpack(
        &self,
        device: DLDevice,
        stream: Option<i64>,
        max_version: DLPackVersion
    ) -> Result<DLPackTensor, Error>;
}

impl From<Box<dyn Array>> for mts_array_t {
    fn from(array: Box<dyn Array>) -> Self {
        // We need to box the box to make sure the pointer is a normal 1-word
        // pointer (`Box<dyn Trait>` contains a 2-words, *fat* pointer which can
        // not be casted to `*mut c_void`)
        let array = Box::new(array);

        return mts_array_t {
            ptr: Box::into_raw(array).cast(),
            origin: Some(rust_array_origin),
            device: Some(rust_array_device),
            dtype: Some(rust_array_dtype),
            as_dlpack: Some(rust_array_as_dlpack),
            shape: Some(rust_array_shape),
            reshape: Some(rust_array_reshape),
            swap_axes: Some(rust_array_swap_axes),
            create: Some(rust_array_create),
            copy: Some(rust_array_copy),
            destroy: Some(rust_array_destroy),
            move_data: Some(rust_array_move_data),
        }
    }
}

macro_rules! check_pointers {
    ($pointer: ident) => {
        if $pointer.is_null() {
            panic!(
                "got invalid NULL pointer for {} at {}:{}",
                stringify!($pointer), file!(), line!()
            );
        }
    };
    ($($pointer: ident),* $(,)?) => {
        $(check_pointers!($pointer);)*
    }
}

pub(super) static RUST_DATA_ORIGIN: Lazy<mts_data_origin_t> = Lazy::new(|| {
    super::origin::register_data_origin("rust.Box<dyn Array>".into()).expect("failed to register a new origin")
});

/******************************************************************************/
/// Implementation of `mts_array_t.origin` using `Box<dyn Array>`
unsafe extern "C" fn rust_array_origin(
    array: *const c_void,
    origin: *mut mts_data_origin_t
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array, origin);
        *origin = *RUST_DATA_ORIGIN;
    })
}

/// Implementation of `mts_array_t.device` using `Box<dyn Array>`
unsafe extern "C" fn rust_array_device(
    array: *const c_void,
    device: *mut DLDevice,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array, device);
        let array = array.cast::<Box<dyn Array>>();
        *device = (*array).device();
    })
}

/// Implementation of `mts_array_t.dtype` using `Box<dyn Array>`
unsafe extern "C" fn rust_array_dtype(
    array: *const c_void,
    dtype: *mut DLDataType,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array, dtype);
        let array = array.cast::<Box<dyn Array>>();
        *dtype = (*array).dtype();
    })
}

/// Implementation of `mts_array_t.shape` using `Box<dyn Array>`
unsafe extern "C" fn rust_array_shape(
    array: *const c_void,
    shape: *mut *const usize,
    shape_count: *mut usize,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array, shape, shape_count);
        let array = array.cast::<Box<dyn Array>>();
        let rust_shape = (*array).shape();

        *shape = rust_shape.as_ptr();
        *shape_count = rust_shape.len();
    })
}

/// Implementation of `mts_array_t.reshape` using `Box<dyn Array>`
#[allow(clippy::cast_possible_truncation)]
unsafe extern "C" fn rust_array_reshape(
    array: *mut c_void,
    shape: *const usize,
    shape_count: usize,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        assert!(shape_count > 0);
        assert!(!shape.is_null());
        check_pointers!(array);
        let array = array.cast::<Box<dyn Array>>();
        let shape = std::slice::from_raw_parts(shape, shape_count);
        (*array).reshape(shape);
    })
}

/// Implementation of `mts_array_t.swap_axes` using `Box<dyn Array>`
#[allow(clippy::cast_possible_truncation)]
unsafe extern "C" fn rust_array_swap_axes(
    array: *mut c_void,
    axis_1: usize,
    axis_2: usize,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array);
        let array = array.cast::<Box<dyn Array>>();
        (*array).swap_axes(axis_1, axis_2);
    })
}

/// Implementation of `mts_array_t.create` using `Box<dyn Array>`
#[allow(clippy::cast_possible_truncation)]
unsafe extern "C" fn rust_array_create(
    array: *const c_void,
    shape: *const usize,
    shape_count: usize,
    array_storage: *mut mts_array_t,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        assert!(shape_count > 0);
        assert!(!shape.is_null());
        check_pointers!(array, shape, array_storage);
        let array = array.cast::<Box<dyn Array>>();

        let shape = std::slice::from_raw_parts(shape, shape_count);
        let new_array = (*array).create(shape);

        *array_storage = new_array.into();
    })
}

/// Implementation of `mts_array_t.copy` using `Box<dyn Array>`
unsafe extern "C" fn rust_array_copy(
    array: *const c_void,
    array_storage: *mut mts_array_t,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array, array_storage);
        let array = array.cast::<Box<dyn Array>>();
        *array_storage = (*array).copy().into();
    })
}

/// Implementation of `mts_array_t.destroy` for `Box<dyn Array>`
unsafe extern "C" fn rust_array_destroy(
    array: *mut c_void,
) {
    if !array.is_null() {
        let array = array.cast::<Box<dyn Array>>();
        let boxed = Box::from_raw(array);
        std::mem::drop(boxed);
    }
}

/// Implementation of `mts_array_t.move_sample` using `Box<dyn Array>`
#[allow(clippy::cast_possible_truncation)]
unsafe extern "C" fn rust_array_move_data(
    output: *mut c_void,
    input: *const c_void,
    movements: *const mts_data_movement_t,
    movements_count: usize,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(output, input);
        let output = output.cast::<Box<dyn Array>>();
        let input = input.cast::<Box<dyn Array>>();

        let movements = if movements_count == 0 {
            &[]
        } else {
            check_pointers!(movements);
            std::slice::from_raw_parts(movements, movements_count)
        };

        (*output).move_data(&**input, movements);
    })
}

/// Implementation of `mts_array_t.as_dlpack` using `Box<dyn Array>`
unsafe extern "C" fn rust_array_as_dlpack(
    array: *mut c_void,
    out: *mut *mut DLManagedTensorVersioned,
    device: DLDevice,
    stream: *const i64,
    max_version: DLPackVersion,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array, out);
        let array = array.cast::<Box<dyn Array>>();
        let stream_opt = stream.as_ref().copied();
        let tensor = (*array).as_dlpack(device, stream_opt, max_version).expect("failed to create DLPack tensor");

        let raw_ptr = tensor.into_raw();
        *out = raw_ptr.as_ptr();
    })
}

impl<T> Array for ndarray::ArcArray<T, ndarray::IxDyn>
where
    T: 'static + Send + Sync + Clone + Default + GetDLPackDataType,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn create(&self, shape: &[usize]) -> Box<dyn Array> {
        return Box::new(ndarray::ArcArray::from_elem(shape, T::default()));
    }

    fn copy(&self) -> Box<dyn Array> {
        return Box::new(self.clone());
    }

    fn shape(&self) -> &[usize] {
        return self.shape();
    }

    fn reshape(&mut self, shape: &[usize]) {
        *self = self.to_shape(shape).expect("invalid shape").to_shared();
    }

    fn swap_axes(&mut self, axis_1: usize, axis_2: usize) {
        self.swap_axes(axis_1, axis_2);
    }

    fn move_data(
        &mut self,
        input: &dyn Array,
        movements: &[mts_data_movement_t],
    ) {
        use ndarray::{Axis, Slice};

        let input = input.as_any().downcast_ref::<ndarray::ArcArray<T, ndarray::IxDyn>>().expect("input must be a ndarray of the same type");

        if movements.is_empty() {
            return;
        }

        // Check if we can use the optimized path (all moves have same property structure)
        let first_prop_start_in = movements[0].properties_start_in;
        let first_prop_start_out = movements[0].properties_start_out;
        let first_prop_len = movements[0].properties_length;

        let mut constant_properties = true;
        let mut contiguous_input_samples = true;
        let mut contiguous_output_samples = true;

        for w in movements.windows(2) {
            if w[0].properties_start_in != first_prop_start_in ||
               w[0].properties_start_out != first_prop_start_out ||
               w[0].properties_length != first_prop_len {
                constant_properties = false;
                break;
            }

            if w[1].sample_in != w[0].sample_in + 1 {
                contiguous_input_samples = false;
            }

            if w[1].sample_out != w[0].sample_out + 1 {
                contiguous_output_samples = false;
            }
        }

        if constant_properties {
            let last = movements.last().unwrap();
            if last.properties_start_in != first_prop_start_in ||
               last.properties_start_out != first_prop_start_out ||
               last.properties_length != first_prop_len {
                constant_properties = false;
            }
        }

        let property_axis = self.shape().len() - 1;

        if constant_properties {
            let input_slice_info = Slice::from(first_prop_start_in..(first_prop_start_in + first_prop_len));
            let output_slice_info = Slice::from(first_prop_start_out..(first_prop_start_out + first_prop_len));

            if contiguous_input_samples && contiguous_output_samples {
                let sample_start_in = movements[0].sample_in;
                let sample_start_out = movements[0].sample_out;
                let sample_count = movements.len();

                let input_samples = input.slice_axis(
                    Axis(0),
                    Slice::from(sample_start_in..(sample_start_in + sample_count))
                );
                let mut output_samples = self.slice_axis_mut(
                    Axis(0),
                    Slice::from(sample_start_out..(sample_start_out + sample_count))
                );

                let value = input_samples.slice_axis(Axis(property_axis), input_slice_info);
                let mut output_location = output_samples.slice_axis_mut(Axis(property_axis), output_slice_info);

                output_location.assign(&value);
            } else {
                for move_item in movements {
                    let input_sample = input.index_axis(Axis(0), move_item.sample_in);
                    let mut output_sample = self.index_axis_mut(Axis(0), move_item.sample_out);

                    let value = input_sample.slice_axis(
                        // property_axis - 1 because we are slicing the sample
                        // axis out, so the property axis is now one less
                        Axis(property_axis - 1),
                        input_slice_info
                    );
                    let mut output_location = output_sample.slice_axis_mut(
                        Axis(property_axis - 1),
                        output_slice_info
                    );
                    output_location.assign(&value);
                }
            }
        } else {
            // fallback to the general case
            for move_item in movements {
                let input_sample = input.index_axis(Axis(0), move_item.sample_in);
                let mut output_sample = self.index_axis_mut(Axis(0), move_item.sample_out);

                let value = input_sample.slice_axis(
                    // see above for property_axis - 1 explanation
                    Axis(property_axis - 1),
                    Slice::from(move_item.properties_start_in..(move_item.properties_start_in + move_item.properties_length))
                );
                let mut output_location = output_sample.slice_axis_mut(
                    Axis(property_axis - 1),
                    Slice::from(move_item.properties_start_out..(move_item.properties_start_out + move_item.properties_length))
                );
                output_location.assign(&value);
            }
        }
    }

    fn device(&self) -> DLDevice {
        DLDevice::cpu()
    }

    fn dtype(&self) -> DLDataType {
        T::get_dlpack_data_type()
    }

    fn as_dlpack(
        &self,
        device: DLDevice,
        stream: Option<i64>,
        max_version: DLPackVersion,
    ) -> Result<DLPackTensor, Error> {
        if stream.is_some() {
            // we only support CPU for now
            return Err(Error {
                code: Some(crate::c_api::MTS_INVALID_PARAMETER_ERROR),
                message: "CPU arrays can not be used with a stream".into(),
            });
        }
        let vendored_version = DLPackVersion::current();
        let major_mismatch = max_version.major != vendored_version.major;
        let minor_too_high = max_version.minor < vendored_version.minor;
        if major_mismatch || minor_too_high {
            return Err(Error {
                code: Some(crate::c_api::MTS_INVALID_PARAMETER_ERROR),
                message: format!(
                    "Metatensor supports DLPack version {}.{}. Caller requested incompatible version {}.{}",
                    vendored_version.major, vendored_version.minor, max_version.major, max_version.minor
                ),
            });
        }

        let ndarray_device = DLDevice::cpu();

        if device.device_type != ndarray_device.device_type || device.device_id != ndarray_device.device_id {
            return Err(Error {
                code: Some(crate::c_api::MTS_INVALID_PARAMETER_ERROR),
                message: format!(
                    "Requested DLPack device ({}) does not match array device ({})",
                    device, ndarray_device
                ),
            });
        }

        let tensor: DLPackTensor = self.try_into().map_err(|e| Error {
            code: Some(crate::c_api::MTS_INVALID_PARAMETER_ERROR),
            message: format!("failed to convert ndarray to DLPack: {:?}", e),
        })?;

        Ok(tensor)
    }
}

/******************************************************************************/

/// An implementation of the [`Array`] trait without any data.
///
/// This only tracks the shape of the array.
#[derive(Debug, Clone)]
pub struct EmptyArray {
    shape: Vec<usize>,
}

impl EmptyArray {
    /// Create a new `EmptyArray` with the given shape.
    pub fn new(shape: Vec<usize>) -> EmptyArray {
        EmptyArray { shape }
    }
}

impl Array for EmptyArray {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn create(&self, shape: &[usize]) -> Box<dyn Array> {
        Box::new(EmptyArray { shape: shape.to_vec() })
    }

    fn copy(&self) -> Box<dyn Array> {
        Box::new(EmptyArray { shape: self.shape.clone() })
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn reshape(&mut self, shape: &[usize]) {
        self.shape = shape.to_vec();
    }

    fn swap_axes(&mut self, axis_1: usize, axis_2: usize) {
        self.shape.swap(axis_1, axis_2);
    }

    fn move_data(&mut self, _: &dyn Array, _: &[mts_data_movement_t]) {
        panic!("can not call Array::move_data() for EmptyArray");
    }

    fn device(&self) -> DLDevice {
        DLDevice::cpu()
    }

    fn dtype(&self) -> DLDataType {
        // Default to f64, consistent with metatensor-core's EmptyDataArray
        DLDataType {
            code: dlpk::sys::DLDataTypeCode::kDLFloat,
            bits: 64,
            lanes: 1,
        }
    }

    fn as_dlpack(
        &self,
        _device: DLDevice,
        _stream: Option<i64>,
        _max_version: DLPackVersion
    ) -> Result<DLPackTensor, Error> {
        panic!("can not call Array::as_dlpack() for EmptyArray");
    }
}

#[cfg(test)]
mod tests {
    use metatensor_sys::{MTS_SUCCESS, mts_array_t};
    use dlpk::sys::{DLDevice, DLManagedTensorVersioned, DLPackVersion, DLDataTypeCode};
    use crate::Array;


    #[test]
    fn ndarray_as_mts_array() {
        let data = ndarray::ArcArray::<f64, _>::zeros(vec![2, 3, 4]);
        let mts_array = mts_array_t::from(Box::new(data) as Box<dyn Array>);

        assert_eq!(mts_array.shape().unwrap(), [2, 3, 4]);

        let created = mts_array.create(&[2, 3, 4]).unwrap();
        assert_eq!(created.shape().unwrap(), [2, 3, 4]);
    }

    #[test]
    fn ndarray_as_mts_array_dlpack() {
        let data = ndarray::ArcArray::<f64, _>::zeros(vec![4, 5, 6]);
        // Wrap it in the C-API struct
        let mts_array = mts_array_t::from(Box::new(data) as Box<dyn Array>);
        unsafe {
            let mut dl_managed: *mut DLManagedTensorVersioned = std::ptr::null_mut();
            let device = DLDevice::cpu();
            // max_version is the consumer's maximum supported version; it must
            // be strictly >= the library's version, so use current + 1 minor
            let current = DLPackVersion::current();
            let max_version = DLPackVersion { major: current.major, minor: current.minor + 1 };
            let status = (mts_array.as_dlpack.unwrap())(
                mts_array.ptr,
                &mut dl_managed,
                device,
                std::ptr::null_mut(),
                max_version
            );

            assert_eq!(status, MTS_SUCCESS);
            assert!(!dl_managed.is_null());
            let tensor = &(*dl_managed).dl_tensor;
            assert_eq!(tensor.ndim, 3);
            assert_eq!(*tensor.shape.offset(0), 4);
            assert_eq!(*tensor.shape.offset(1), 5);
            assert_eq!(*tensor.shape.offset(2), 6);
            // The dtype for f64 is kDLFloat (2) with 64 bits
            assert_eq!(tensor.dtype.code, DLDataTypeCode::kDLFloat);
            assert_eq!(tensor.dtype.bits, 64);
            // Clean up using the provided deleter
            if let Some(deleter) = (*dl_managed).deleter {
                deleter(dl_managed);
            }
        }
    }

    #[test]
    fn ndarray_generic_support() {
        let data = ndarray::ArcArray::<i32, _>::from_elem(vec![2, 2], 42);
        let mts_array = mts_array_t::from(Box::new(data) as Box<dyn Array>);

        assert_eq!(mts_array.shape().unwrap(), [2, 2]);

        // Should be able to export as DLPack
        unsafe {
            let mut dl_managed: *mut DLManagedTensorVersioned = std::ptr::null_mut();
            let status = (mts_array.as_dlpack.unwrap())(
                mts_array.ptr,
                &mut dl_managed,
                DLDevice::cpu(),
                std::ptr::null_mut(),
                DLPackVersion::current()
            );
            assert_eq!(status, MTS_SUCCESS);

            let tensor = &(*dl_managed).dl_tensor;
            assert_eq!(tensor.dtype.code, DLDataTypeCode::kDLInt);
            assert_eq!(tensor.dtype.bits, 32);

            if let Some(deleter) = (*dl_managed).deleter {
                deleter(dl_managed);
            }
        }

        // And creation should make an array of the same type (i32)
        let created = mts_array.create(&[1, 1]).unwrap();
        unsafe {
            let mut dl_managed: *mut DLManagedTensorVersioned = std::ptr::null_mut();
            let status = (created.as_dlpack.unwrap())(
                created.ptr,
                &mut dl_managed,
                DLDevice::cpu(),
                std::ptr::null_mut(),
                DLPackVersion::current()
            );
            assert_eq!(status, MTS_SUCCESS);
            let tensor = &(*dl_managed).dl_tensor;
            assert_eq!(tensor.dtype.code, DLDataTypeCode::kDLInt);

             if let Some(deleter) = (*dl_managed).deleter {
                deleter(dl_managed);
            }
        }
    }

    #[test]
    fn ndarray_device() {
        let data = ndarray::ArcArray::<f64, _>::zeros(vec![2, 3]);
        let mts_array = mts_array_t::from(Box::new(data) as Box<dyn Array>);

        // Test via the device function pointer
        unsafe {
            let device_fn = mts_array.device.expect("device function should be set");
            let mut device = DLDevice::cpu();
            device.device_id = 99; // sentinel to confirm it's overwritten
            let status = device_fn(mts_array.ptr, &mut device);
            assert_eq!(status, MTS_SUCCESS);
            assert_eq!(device.device_type, DLDevice::cpu().device_type);
            assert_eq!(device.device_id, 0);
        }
    }

    #[test]
    fn empty_array_device() {
        use crate::data::EmptyArray;
        let arr = EmptyArray::new(vec![2, 3]);
        let dev = arr.device();
        assert_eq!(dev.device_type, DLDevice::cpu().device_type);
        assert_eq!(dev.device_id, 0);
    }
}
