use std::os::raw::c_void;

use once_cell::sync::Lazy;

use dlpk::sys::{DLDevice, DLManagedTensorVersioned, DLPackVersion, DLDataType};
use dlpk::DLPackTensor;

use crate::errors::Error;
use crate::c_api::{mts_array_t, mts_data_origin_t, mts_data_movement_t, mts_status_t};

use super::MtsArray;

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

    /// Create a new array with the same array origin, data type, and device as
    /// the current one, but with the requested `shape`.
    ///
    /// The new array should be filled with the scalar value from `fill_value`,
    /// which must be an `MtsArray` with shape `(1,)` and the same dtype as this
    /// array.
    fn create(&self, shape: &[usize], fill_value: MtsArray) -> Box<dyn Array>;

    /// Make a copy of this `array`
    ///
    /// The new array is expected to have the same array origin and data type,
    /// but live on the given device.
    fn copy(&self, device: DLDevice) -> Box<dyn Array>;

    /// Get the shape of the array. This can be empty if the array has no shape
    /// (e.g. a scalar).
    fn shape(&self) -> Vec<usize>;

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

pub (super) struct RustArray {
    impl_: Box<dyn Array>,
    shape: Vec<usize>,
}

impl std::ops::Deref for RustArray {
    type Target = dyn Array;

    fn deref(&self) -> &Self::Target {
        &*self.impl_
    }
}

impl std::ops::DerefMut for RustArray {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.impl_
    }
}

impl From<Box<dyn Array>> for MtsArray {
    fn from(value: Box<dyn Array>) -> Self {
        let shape = value.shape();
        let array = RustArray {
            impl_: value,
            shape,
        };

        let raw = mts_array_t {
            ptr: Box::into_raw(Box::new(array)).cast(),
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
        };

        return MtsArray::from_raw(raw);
    }
}

impl<T> From<T> for MtsArray where T: Array + 'static {
    fn from(value: T) -> Self {
        let boxed = Box::new(value) as Box<dyn Array>;
        return MtsArray::from(boxed);
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
    super::origin::register_data_origin("RustArray".into()).expect("failed to register a new origin")
});

/******************************************************************************/
/// Implementation of `mts_array_t.origin` using `RustArray`
unsafe extern "C" fn rust_array_origin(
    array: *const c_void,
    origin: *mut mts_data_origin_t
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array, origin);
        *origin = *RUST_DATA_ORIGIN;

        Ok(())
    })
}

/// Implementation of `mts_array_t.device` using `RustArray`
unsafe extern "C" fn rust_array_device(
    array: *const c_void,
    device: *mut DLDevice,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array, device);
        let array = array.cast::<RustArray>();
        *device = (*array).impl_.device();

        Ok(())
    })
}

/// Implementation of `mts_array_t.dtype` using `RustArray`
unsafe extern "C" fn rust_array_dtype(
    array: *const c_void,
    dtype: *mut DLDataType,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array, dtype);
        let array = array.cast::<RustArray>();
        *dtype = (*array).impl_.dtype();

        Ok(())
    })
}

/// Implementation of `mts_array_t.shape` using `RustArray`
unsafe extern "C" fn rust_array_shape(
    array: *const c_void,
    shape: *mut *const usize,
    shape_count: *mut usize,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array, shape, shape_count);
        let array = array.cast::<RustArray>();
        let rust_shape = &(*array).shape;

        *shape = rust_shape.as_ptr();
        *shape_count = rust_shape.len();

        Ok(())
    })
}

/// Implementation of `mts_array_t.reshape` using `RustArray`
#[allow(clippy::cast_possible_truncation)]
unsafe extern "C" fn rust_array_reshape(
    array: *mut c_void,
    shape: *const usize,
    shape_count: usize,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array);
        let array = array.cast::<RustArray>();

        let shape = if shape_count == 0 {
            &[]
        } else {
            check_pointers!(shape);
            std::slice::from_raw_parts(shape, shape_count)
        };

        (*array).impl_.reshape(shape);
        (*array).shape = shape.to_vec();

        Ok(())
    })
}

/// Implementation of `mts_array_t.swap_axes` using `RustArray`
#[allow(clippy::cast_possible_truncation)]
unsafe extern "C" fn rust_array_swap_axes(
    array: *mut c_void,
    axis_1: usize,
    axis_2: usize,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array);
        let array = array.cast::<RustArray>();
        (*array).impl_.swap_axes(axis_1, axis_2);
        (*array).shape.swap(axis_1, axis_2);

        Ok(())
    })
}

/// Implementation of `mts_array_t.create` using `RustArray`
#[allow(clippy::cast_possible_truncation)]
unsafe extern "C" fn rust_array_create(
    array: *const c_void,
    shape: *const usize,
    shape_count: usize,
    fill_value: mts_array_t,
    array_storage: *mut mts_array_t,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array, array_storage);
        let array = array.cast::<RustArray>();

        let shape = if shape_count == 0 {
            &[]
        } else {
            check_pointers!(shape);
            std::slice::from_raw_parts(shape, shape_count)
        };

        let new_array = (*array).impl_.create(shape, MtsArray::from_raw(fill_value));
        let new_array = MtsArray::from(new_array);

        *array_storage = new_array.into_raw();

        Ok(())
    })
}

/// Implementation of `mts_array_t.copy` using `RustArray`
unsafe extern "C" fn rust_array_copy(
    array: *const c_void,
    device: DLDevice,
    new_array: *mut mts_array_t
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array, new_array);
        let array = array.cast::<RustArray>();

        let copy = (*array).impl_.copy(device);
        let copy = MtsArray::from(copy);
        *new_array = copy.into_raw();

        Ok(())
    })
}

/// Implementation of `mts_array_t.destroy` for `RustArray`
unsafe extern "C" fn rust_array_destroy(
    array: *mut c_void,
) {
    if !array.is_null() {
        let array = array.cast::<RustArray>();
        let boxed = Box::from_raw(array);
        std::mem::drop(boxed);
    }
}

/// Implementation of `mts_array_t.move_sample` using `RustArray`
#[allow(clippy::cast_possible_truncation)]
unsafe extern "C" fn rust_array_move_data(
    output: *mut c_void,
    input: *const c_void,
    movements: *const mts_data_movement_t,
    movements_count: usize,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(output, input);
        let output = output.cast::<RustArray>();
        let input = input.cast::<RustArray>();

        let movements = if movements_count == 0 {
            &[]
        } else {
            check_pointers!(movements);
            std::slice::from_raw_parts(movements, movements_count)
        };

        (*output).impl_.move_data(&*(*input).impl_, movements);

        Ok(())
    })
}

/// Implementation of `mts_array_t.as_dlpack` using `RustArray`
unsafe extern "C" fn rust_array_as_dlpack(
    array: *mut c_void,
    out: *mut *mut DLManagedTensorVersioned,
    device: DLDevice,
    stream: *const i64,
    max_version: DLPackVersion,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array, out);
        let array = array.cast::<RustArray>();
        let stream_opt = stream.as_ref().copied();
        let tensor = (*array).impl_.as_dlpack(device, stream_opt, max_version)?;

        *out = tensor.into_raw().as_ptr();

        Ok(())
    })
}
