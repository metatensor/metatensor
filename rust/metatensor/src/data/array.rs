use std::ops::Range;
use std::os::raw::c_void;
use once_cell::sync::Lazy;

use crate::c_api::{mts_array_t, mts_data_origin_t, mts_sample_mapping_t, mts_status_t};

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

    /// Get the underlying data storage as a contiguous slice
    ///
    /// This function is allowed to panic if the data is not accessible in RAM,
    /// not stored as 64-bit floating point values, or not stored as a
    /// C-contiguous array.
    fn data(&mut self) -> &mut [f64];

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
    /// The `samples` indicate where the data should be moved from `input` to
    /// `output`.
    ///
    /// This function should copy data from `input[sample.input, ..., :]` to
    /// `array[sample.output, ..., properties]` for each sample in `samples`.
    /// All indexes are 0-based.
    fn move_samples_from(
        &mut self,
        input: &dyn Array,
        samples: &[mts_sample_mapping_t],
        properties: Range<usize>,
    );

    /// Get a DLPack view of this array.
    /// 
    /// Returns None if the array doesn't support DLPack conversion.
    fn as_dlpack(&self) -> Option<*mut metatensor_sys::DLManagedTensorVersioned> {
        None  // Default implementation returns None
    }
}

impl From<Box<dyn Array>> for mts_array_t {
    fn from(array: Box<dyn Array>) -> Self {
        // We need to box the box to make sure the pointer is a normal 1-word
        // pointer (`Box<dyn Trait>` contains a 2-words, *fat* pointer which can
        // not be casted to `*mut c_void`)
        let array = Box::new(array);

        mts_array_t {
            ptr: Box::into_raw(array).cast(),
            origin: Some(rust_array_origin),
            data: Some(rust_array_data),
            shape: Some(rust_array_shape),
            reshape: Some(rust_array_reshape),
            swap_axes: Some(rust_array_swap_axes),
            create: Some(rust_array_create),
            copy: Some(rust_array_copy),
            destroy: Some(rust_array_destroy),
            move_samples_from: Some(rust_array_move_samples_from),
            as_dlpack: Some(rust_array_as_dlpack),
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

/// Implementation of `mts_array_t.data` for `Box<dyn Array>`
unsafe extern "C" fn rust_array_data(
    array: *mut c_void,
    data: *mut *mut f64,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array, data);
        let array = array.cast::<Box<dyn Array>>();
        *data = (*array).data().as_mut_ptr();
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
unsafe extern "C" fn rust_array_move_samples_from(
    output: *mut c_void,
    input: *const c_void,
    samples: *const mts_sample_mapping_t,
    samples_count: usize,
    property_start: usize,
    property_end: usize,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(output, input);
        let output = output.cast::<Box<dyn Array>>();
        let input = input.cast::<Box<dyn Array>>();

        let samples = if samples_count == 0 {
            &[]
        } else {
            check_pointers!(samples);
            std::slice::from_raw_parts(samples, samples_count)
        };

        (*output).move_samples_from(&**input, samples, property_start..property_end);
    })
}

/// Implementation of `mts_array_t.as_dlpack` using `Box<dyn Array>`
unsafe extern "C" fn rust_array_as_dlpack(
    array: *const c_void,
    dl_managed_tensor: *mut *mut metatensor_sys::DLManagedTensorVersioned,
) -> mts_status_t {
    crate::errors::catch_unwind(|| {
        check_pointers!(array, dl_managed_tensor);
        let array = array.cast::<Box<dyn Array>>();
        
        match (*array).as_dlpack() {
            Some(tensor) => {
                *dl_managed_tensor = tensor;
            },
            None => {
                panic!("DLPack conversion not implemented for this array type");
            }
        }
    })
}

/******************************************************************************/

impl Array for ndarray::ArrayD<f64> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn create(&self, shape: &[usize]) -> Box<dyn Array> {
        return Box::new(ndarray::Array::from_elem(shape, 0.0));
    }

    fn copy(&self) -> Box<dyn Array> {
        return Box::new(self.clone());
    }

    fn data(&mut self) -> &mut [f64] {
        return self.as_slice_mut().expect("array is not contiguous")
    }

    fn shape(&self) -> &[usize] {
        return self.shape();
    }

    fn reshape(&mut self, shape: &[usize]) {
        let mut array = std::mem::take(self);
        array = array.to_shape(shape).expect("invalid shape").to_owned();
        std::mem::swap(self, &mut array);
    }

    fn swap_axes(&mut self, axis_1: usize, axis_2: usize) {
        self.swap_axes(axis_1, axis_2);
    }

    fn move_samples_from(
        &mut self,
        input: &dyn Array,
        samples: &[mts_sample_mapping_t],
        property: Range<usize>,
    ) {
        use ndarray::{Axis, Slice};

        // -2 since we also remove one axis with `index_axis_mut` below
        let property_axis = self.shape().len() - 2;

        let input = input.as_any().downcast_ref::<ndarray::ArrayD<f64>>().expect("input must be a ndarray");
        for sample in samples {
            let value = input.index_axis(Axis(0), sample.input);

            let mut output_location = self.index_axis_mut(Axis(0), sample.output);
            let mut output_location = output_location.slice_axis_mut(
                Axis(property_axis), Slice::from(property.clone())
            );

            output_location.assign(&value);
        }
    }

    fn as_dlpack(&self) -> Option<*mut metatensor_sys::DLManagedTensorVersioned> {
        use dlpack::DLPackTensor;
        let array_clone = self.clone();
        let result: Result<DLPackTensor, _> = array_clone.try_into();
        match result {
            Ok(dlpack_tensor) => {
                let ptr = dlpack_tensor.into_raw() as *mut metatensor_sys::DLManagedTensorVersioned;
                Some(ptr)
            },
            Err(_) => None,
        }
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

    fn data(&mut self) -> &mut [f64] {
        panic!("can not call Array::data() for EmptyArray");
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

    fn move_samples_from(&mut self, _: &dyn Array, _: &[mts_sample_mapping_t], _: Range<usize>) {
        panic!("can not call Array::move_samples_from() for EmptyArray");
    }

    fn as_dlpack(&self) -> Option<*mut metatensor_sys::DLManagedTensorVersioned> {
        panic!("can not call Array::as_dlpack() for EmptyArray");
    }
}

#[cfg(test)]
mod tests {
    use metatensor_sys::*;

    use super::*;

    #[test]
    fn ndarray_as_mts_array() {
        let data = ndarray::ArrayD::<f64>::zeros(vec![4, 5, 6]);

        let address = data.as_ptr() as usize;
        let mut mts_array = mts_array_t::from(Box::new(data) as Box<dyn Array>);

        let mut rust_origin = 0;
        let status = unsafe {
            mts_register_data_origin(b"rust.Box<dyn Array>\0".as_ptr().cast(), &mut rust_origin)
        };
        assert_eq!(status, MTS_SUCCESS);

        assert_eq!(mts_array.origin().unwrap(), rust_origin);

        assert_eq!(mts_array.shape().unwrap(), [4, 5, 6]);

        assert_eq!(mts_array.data().unwrap().as_ptr() as usize, address);

        mts_array.reshape(&[20, 6, 1]).unwrap();
        assert_eq!(mts_array.shape().unwrap(), [20, 6, 1]);

        mts_array.swap_axes(1, 2).unwrap();
        assert_eq!(mts_array.shape().unwrap(), [20, 1, 6]);

        let copy = mts_array.copy().unwrap();
        assert_eq!(copy.origin().unwrap(), rust_origin);
        assert_eq!(copy.shape().unwrap(), [20, 1, 6]);
        assert_ne!(mts_array.data().unwrap().as_ptr() as usize, address);

        let created = mts_array.create(&[2, 3, 4]).unwrap();
        assert_eq!(created.origin().unwrap(), rust_origin);
        assert_eq!(created.shape().unwrap(), [2, 3, 4]);
        assert_ne!(mts_array.data().unwrap().as_ptr() as usize, address);
    }
}
