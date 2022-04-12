use std::os::raw::c_void;
use std::sync::Mutex;
use std::ops::Range;

use once_cell::sync::Lazy;

use crate::c_api::{aml_status_t, catch_unwind};
use crate::{Error, check_pointers};

/// A single 64-bit integer representing a data origin (numpy ndarray, rust
/// ndarray, torch tensor, fortran array, ...).
#[repr(transparent)]
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct aml_data_origin_t(pub u64);

pub type DataOrigin = aml_data_origin_t;

static REGISTERED_DATA_ORIGIN: Lazy<Mutex<Vec<String>>> = Lazy::new(|| {
    // start the registered origins at 1, this allow using 0 as a marker for
    // "unknown data origin"
    Mutex::new(vec!["unregistered origin".into()])
});

/// Register a new data origin with the given `name`, or get the
/// `DataOrigin` corresponding to this name if it was already registered.
pub fn register_data_origin(name: String) -> DataOrigin {
    let mut registered_origins = REGISTERED_DATA_ORIGIN.lock().expect("mutex got poisoned");

    for (i, registered) in registered_origins.iter().enumerate() {
        if registered == &name {
            return aml_data_origin_t(i as u64);
        }
    }

    // could not find the origin, register a new one
    registered_origins.push(name);

    return aml_data_origin_t((registered_origins.len() - 1) as u64);
}

/// Get the name of the given (pre-registered) origin
#[allow(clippy::cast_possible_truncation)]
pub fn get_data_origin(origin: DataOrigin) -> String {
    let registered_origins = REGISTERED_DATA_ORIGIN.lock().expect("mutex got poisoned");
    let id = origin.0 as usize;
    if id < registered_origins.len() {
        return registered_origins[id].clone();
    } else {
        return registered_origins[0].clone();
    }
}

/// `aml_array_t` manages 3D arrays the be used as data in a block/tensor map.
/// The array itself if opaque to this library and can come from multiple
/// sources: Rust program, a C/C++ program, a Fortran program, Python with numpy
/// or torch. The data does not have to live on CPU, or even on the same machine
/// where this code is executed.
///
/// This struct contains a C-compatible manual implementation of a virtual table
/// (vtable, i.e. trait in Rust, pure virtual class in C++); allowing
/// manipulation of the array in an opaque way.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct aml_array_t {
    /// User-provided data should be stored here, it will be passed as the
    /// first parameter to all function pointers below.
    pub ptr: *mut c_void,

    /// This function needs to store the "data origin" for this array in
    /// `origin`. Users of `aml_array_t` should register a single data
    /// origin with `register_data_origin`, and use it for all compatible
    /// arrays.
    origin: Option<unsafe extern fn(
        array: *const c_void,
        origin: *mut aml_data_origin_t
    ) -> aml_status_t>,

    /// Get the shape of the array managed by this `aml_array_t` in the `*shape`
    /// pointer, and the number of dimension (size of the `*shape` array) in
    /// `*shape_count`.
    shape: Option<unsafe extern fn(
        array: *const c_void,
        shape: *mut *const usize,
        shape_count: *mut usize,
    ) -> aml_status_t>,

    /// Change the shape of the array managed by this `aml_array_t` to the given
    /// `shape`. `shape_count` must contain the number of elements in the
    /// `shape` array
    reshape: Option<unsafe extern fn(
        array: *mut c_void,
        shape: *const usize,
        shape_count: usize,
    ) -> aml_status_t>,

    /// Swap the axes `axis_1` and `axis_2` in this `array`.
    swap_axes: Option<unsafe extern fn(
        array: *mut c_void,
        axis_1: usize,
        axis_2: usize,
    ) -> aml_status_t>,

    /// Create a new array with the same options as the current one (data type,
    /// data location, etc.) and the requested `shape`; and store it in
    /// `new_array`. The number of elements in the `shape` array should be given
    /// in `shape_count`.
    ///
    /// The new array should be filled with zeros.
    create: Option<unsafe extern fn(
        array: *const c_void,
        shape: *const usize,
        shape_count: usize,
        new_array: *mut aml_array_t,
    ) -> aml_status_t>,

    /// Make a copy of this `array` and return the new array in `new_array`
    copy: Option<unsafe extern fn(
        array: *const c_void,
        new_array: *mut aml_array_t,
    ) -> aml_status_t>,

    /// Remove this array and free the associated memory. This function can be
    /// set to `NULL` is there is no memory management to do.
    destroy: Option<unsafe extern fn(array: *mut c_void)>,

    /// Set entries in this array taking data from the `other_array`. This array
    /// is guaranteed to be created by calling `aml_array_t::create` with one of
    /// the arrays in the same block or tensor map as this `array`.
    ///
    /// This function should copy data from `other_array[other_sample, ..., :]` to
    /// `array[sample, ..., property_start:property_end]`. All indexes are 0-based.
    move_sample: Option<unsafe extern fn(
        array: *mut c_void,
        sample: u64,
        property_start: u64,
        property_end: u64,
        other_array: *const c_void,
        other_sample: u64
    ) -> aml_status_t>,
}

impl std::fmt::Debug for aml_array_t {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut origin = None;
        let mut shape = None;
        if !self.ptr.is_null() {
            origin = self.origin().ok().map(get_data_origin);
            shape = self.shape().ok();
        }

        f.debug_struct("aml_array_t")
            .field("ptr", &self.ptr)
            .field("origin", &origin)
            .field("shape", &shape)
            .finish()
    }
}

impl Drop for aml_array_t {
    fn drop(&mut self) {
        if let Some(function) = self.destroy {
            unsafe { function(self.ptr) }
        }
    }
}

impl Clone for aml_array_t {
    fn clone(&self) -> aml_array_t {
        if let Some(function) = self.copy {
            let mut new_array = aml_array_t::null();
            let status  = unsafe { function(self.ptr, &mut new_array) };
            assert!(status.is_success(), "calling aml_array_t.copy failed");
            return new_array;
        } else {
            panic!("function aml_array_t.copy is not set")
        }
    }
}

impl aml_array_t {
    /// Create an `aml_array_t` from a Rust implementation of the `DataStorage`
    /// trait.
    pub fn new(value: Box<dyn DataStorage>) -> aml_array_t {
        // We need to box the box to make sure the pointer is a normal 1-word
        // pointer (`Box<dyn Trait>` contains a 2-words, *fat* pointer which can
        // not be casted to `*mut c_void`)
        let array = Box::new(value);
        aml_array_t {
            ptr: Box::into_raw(array).cast(),
            origin: Some(rust_data_origin),
            shape: Some(rust_data_shape),
            reshape: Some(rust_data_reshape),
            swap_axes: Some(rust_data_swap_axes),
            create: Some(rust_data_create),
            copy: Some(rust_data_copy),
            destroy: Some(rust_data_destroy),
            move_sample: Some(rust_data_move_sample),
        }
    }

    /// make a raw (member by member) copy of the array. Contrary to
    /// `aml_array_t::clone`, the returned array refers to the same
    /// `aml_array_t` instance, and as such should not be freed.
    pub(crate) fn raw_copy(&self) -> aml_array_t {
        aml_array_t {
            ptr: self.ptr,
            origin: self.origin,
            shape: self.shape,
            reshape: self.reshape,
            swap_axes: self.swap_axes,
            create: self.create,
            copy: self.copy,
            destroy: None,
            move_sample: self.move_sample,
        }
    }

    /// Create an `aml_array_t` with all fields set to null pointers.
    fn null() -> aml_array_t {
        aml_array_t {
            ptr: std::ptr::null_mut(),
            origin: None,
            shape: None,
            reshape: None,
            swap_axes: None,
            create: None,
            copy: None,
            destroy: None,
            move_sample: None,
        }
    }

    /// Get the origin of this array
    pub fn origin(&self) -> Result<aml_data_origin_t, Error> {
        let function = self.origin.expect("aml_array_t.origin function is NULL");

        let mut origin = aml_data_origin_t(0);
        let status = unsafe {
            function(self.ptr, &mut origin)
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling aml_array_t.origin failed".into()
            });
        }

        return Ok(origin);
    }

    /// Get the shape of this array
    #[allow(clippy::cast_possible_truncation)]
    pub fn shape(&self) -> Result<&[usize], Error> {
        let function = self.shape.expect("aml_array_t.shape function is NULL");

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
                status, context: "calling aml_array_t.shape failed".into()
            });
        }

        let shape = unsafe {
            std::slice::from_raw_parts(shape, shape_count)
        };

        return Ok(shape);
    }

    /// Set the shape of this array to the given new `shape`
    pub fn reshape(&mut self, shape: &[usize]) -> Result<(), Error> {
        let function = self.reshape.expect("aml_array_t.reshape function is NULL");

        let status = unsafe {
            function(
                self.ptr,
                shape.as_ptr(),
                shape.len(),
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling aml_array_t.reshape failed".into()
            });
        }

        return Ok(());
    }

    /// Swap the axes `axis_1` and `axis_2` in the dimensions of this array.
    pub fn swap_axes(&mut self, axis_1: usize, axis_2: usize) -> Result<(), Error> {
        let function = self.swap_axes.expect("aml_array_t.swap_axes function is NULL");

        let status = unsafe {
            function(
                self.ptr,
                axis_1,
                axis_2,
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling aml_array_t.swap_axes failed".into()
            });
        }

        return Ok(());
    }

    /// Create a new array with the same settings as this one and the given `shape`
    pub fn create(&self, shape: &[usize]) -> Result<aml_array_t, Error> {
        let function = self.create.expect("aml_array_t.create function is NULL");

        let mut data_storage = aml_array_t::null();
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
                status, context: "calling aml_array_t.create failed".into()
            });
        }

        return Ok(data_storage);
    }

    /// Set entries in this array taking data from the `other` array. This array
    /// MUST have been created by calling `aml_array_t.create()` with one of the
    /// other arrays in the same block or tensor map.
    ///
    /// This function will copy data from `other_array[other_sample, ..., :]` to
    /// `array[sample, ..., property_start:property_end]`.
    pub fn move_sample(
        &mut self,
        sample: usize,
        properties: std::ops::Range<usize>,
        other: &aml_array_t,
        other_sample: usize
    ) -> Result<(), Error> {
        let function = self.move_sample.expect("aml_array_t.move_sample function is NULL");

        let status = unsafe {
            function(
                self.ptr,
                sample as u64,
                properties.start as u64,
                properties.end as u64,
                other.ptr,
                other_sample as u64,
            )
        };

        if !status.is_success() {
            return Err(Error::External {
                status, context: "calling aml_array_t.move_sample failed".into()
            });
        }

        return Ok(());
    }

    /// Get the data in this `aml_array_t` as a `ndarray::ArrayD`. This function
    /// will panic if the data in this `aml_array_t` is not a `ndarray::ArrayD`.
    #[cfg(feature = "ndarray")]
    pub fn as_array(&self) -> &ndarray::ArrayD<f64> {
        assert_eq!(
            self.origin().unwrap_or(aml_data_origin_t(0)), *NDARRAY_DATA_ORIGIN,
            "this array was not create by rust ndarray"
        );

        let array = self.ptr.cast::<Box<dyn DataStorage>>();
        unsafe {
            (*array).as_any().downcast_ref().expect("invalid array type")
        }
    }

    #[cfg(feature = "ndarray")]
    pub fn as_array_mut(&mut self) -> &mut ndarray::Array3<f64> {
        assert_eq!(
            self.origin().unwrap_or(aml_data_origin_t(0)), *NDARRAY_DATA_ORIGIN,
            "this array was not create by rust ndarray"
        );

        let array = self.ptr.cast::<Box<dyn DataStorage>>();
        unsafe {
            (*array).as_any_mut().downcast_mut().expect("invalid array type")
        }
    }
}

/// A rust trait with the same interface as `aml_array_t`, see this struct for
/// the documentation
pub trait DataStorage: std::any::Any{
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    fn origin(&self) -> DataOrigin;

    fn create(&self, shape: &[usize]) -> Box<dyn DataStorage>;

    fn copy(&self) -> Box<dyn DataStorage>;

    fn shape(&self) -> &[usize];
    fn reshape(&mut self, shape: &[usize]);
    fn swap_axes(&mut self, axis_1: usize, axis_2: usize);

    fn move_sample(
        &mut self,
        sample: usize,
        properties: Range<usize>,
        other: &dyn DataStorage,
        sample_other: usize
    );
}

/// Implementation of `aml_array_t.origin` using `Box<dyn DataStorage>`
unsafe extern fn rust_data_origin(
    data: *const c_void,
    origin: *mut aml_data_origin_t
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(data, origin);
        let data = data.cast::<Box<dyn DataStorage>>();

        *origin = (*data).origin();
        Ok(())
    })
}

/// Implementation of `aml_array_t.shape` using `Box<dyn DataStorage>`
unsafe extern fn rust_data_shape(
    data: *const c_void,
    shape: *mut *const usize,
    shape_count: *mut usize,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(data, shape, shape_count);
        let data = data.cast::<Box<dyn DataStorage>>();
        let rust_shape = (*data).shape();

        *shape = rust_shape.as_ptr();
        *shape_count = rust_shape.len();

        Ok(())
    })
}

/// Implementation of `aml_array_t.reshape` using `Box<dyn DataStorage>`
#[allow(clippy::cast_possible_truncation)]
unsafe extern fn rust_data_reshape(
    data: *mut c_void,
    shape: *const usize,
    shape_count: usize,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(data);
        let data = data.cast::<Box<dyn DataStorage>>();
        let shape = std::slice::from_raw_parts(shape, shape_count);
        (*data).reshape(shape);
        Ok(())
    })
}

/// Implementation of `aml_array_t.swap_axes` using `Box<dyn DataStorage>`
#[allow(clippy::cast_possible_truncation)]
unsafe extern fn rust_data_swap_axes(
    data: *mut c_void,
    axis_1: usize,
    axis_2: usize,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(data);
        let data = data.cast::<Box<dyn DataStorage>>();
        (*data).swap_axes(axis_1, axis_2);
        Ok(())
    })
}

/// Implementation of `aml_array_t.create` using `Box<dyn DataStorage>`
#[allow(clippy::cast_possible_truncation)]
unsafe extern fn rust_data_create(
    data: *const c_void,
    shape: *const usize,
    shape_count: usize,
    data_storage: *mut aml_array_t,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(data, data_storage);
        let data = data.cast::<Box<dyn DataStorage>>();

        let shape = std::slice::from_raw_parts(shape, shape_count);
        let new_data = (*data).create(shape);

        *data_storage = aml_array_t::new(new_data);

        Ok(())
    })
}


/// Implementation of `aml_array_t.copy` using `Box<dyn DataStorage>`
unsafe extern fn rust_data_copy(
    data: *const c_void,
    data_storage: *mut aml_array_t,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(data, data_storage);
        let data = data.cast::<Box<dyn DataStorage>>();
        *data_storage = aml_array_t::new((*data).copy());

        Ok(())
    })
}

/// Implementation of `aml_array_t.destroy` for `Box<dyn DataStorage>`
unsafe extern fn rust_data_destroy(
    data: *mut c_void,
) {
    if !data.is_null() {
        let data = data.cast::<Box<dyn DataStorage>>();
        let boxed = Box::from_raw(data);
        std::mem::drop(boxed);
    }
}

/// Implementation of `aml_array_t.move_sample` using `Box<dyn DataStorage>`
#[allow(clippy::cast_possible_truncation)]
unsafe extern fn rust_data_move_sample(
    data: *mut c_void,
    sample: u64,
    property_start: u64,
    property_end: u64,
    other: *const c_void,
    other_sample: u64
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(data, other);
        let data = data.cast::<Box<dyn DataStorage>>();
        let other = other.cast::<Box<dyn DataStorage>>();

        (*data).move_sample(
            sample as usize,
            property_start as usize .. property_end as usize,
            &**other,
            other_sample as usize,
        );

        Ok(())
    })
}

/******************************************************************************/

#[cfg(feature = "ndarray")]
static NDARRAY_DATA_ORIGIN: Lazy<DataOrigin> = Lazy::new(|| {
    register_data_origin("rust.ndarray".into())
});


#[cfg(feature = "ndarray")]
impl DataStorage for ndarray::ArrayD<f64> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn origin(&self) -> DataOrigin {
        return *NDARRAY_DATA_ORIGIN;
    }

    fn create(&self, shape: &[usize]) -> Box<dyn DataStorage> {
        return Box::new(ndarray::Array::from_elem(shape, 0.0));
    }

    fn copy(&self) -> Box<dyn DataStorage> {
        return Box::new(self.clone());
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

    fn move_sample(
        &mut self,
        sample: usize,
        properties: Range<usize>,
        other: &dyn DataStorage,
        sample_other: usize
    ) {
        use ndarray::{Axis, Slice};

        let other = other.as_any().downcast_ref::<ndarray::ArrayD<f64>>().expect("other must be a ndarray");
        let value = other.index_axis(Axis(0), sample_other);

        // -2 since we also remove one axis with `index_axis_mut`
        let property_axis = self.shape().len() - 2;
        let mut output = self.index_axis_mut(Axis(0), sample);
        let mut output = output.slice_axis_mut(Axis(property_axis), Slice::from(properties));

        output.assign(&value);
    }
}

/******************************************************************************/

#[cfg(test)]
pub use self::tests::TestArray;

#[cfg(test)]
mod tests {
    use super::*;

    static DUMMY_DATA_ORIGIN: Lazy<DataOrigin> = Lazy::new(|| {
        register_data_origin("dummy test data".into())
    });

    /// An implementation of the `DataStorage` trait without any data. This only
    /// tracks the shape of the array.
    pub struct TestArray {
        shape: Vec<usize>,
    }

    impl TestArray {
        pub fn new(shape: Vec<usize>) -> TestArray {
            TestArray { shape }
        }
    }

    impl DataStorage for TestArray {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }

        fn origin(&self) -> crate::DataOrigin {
            *DUMMY_DATA_ORIGIN
        }

        fn create(&self, shape: &[usize]) -> Box<dyn DataStorage> {
            Box::new(TestArray { shape: shape.to_vec() })
        }

        fn copy(&self) -> Box<dyn DataStorage> {
            Box::new(TestArray { shape: self.shape.clone() })
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

        fn move_sample(
            &mut self,
            _sample: usize,
            _properties: std::ops::Range<usize>,
            _other: &dyn DataStorage,
            _sample_other: usize
        ) {
            unimplemented!()
        }
    }

    #[test]
    fn data_origin() {
        assert_eq!(get_data_origin(aml_data_origin_t(0)), "unregistered origin");
        assert_eq!(get_data_origin(aml_data_origin_t(10000)), "unregistered origin");

        let origin = register_data_origin("test origin".into());
        assert_eq!(get_data_origin(origin), "test origin");
    }

    #[test]
    fn debug() {
        let data = aml_array_t::new(Box::new(TestArray::new(vec![3, 4, 5])));

        let debug_format = format!("{:?}", data);
        assert_eq!(debug_format, format!(
            "aml_array_t {{ ptr: {:?}, origin: Some(\"dummy test data\"), shape: Some([3, 4, 5]) }}",
            data.ptr
        ));
    }

    #[cfg(feature = "ndarray")]
    mod ndarray {
        use ndarray::ArrayD;

        use crate::{aml_array_t, get_data_origin};

        #[test]
        fn shape() {
            let mut data = aml_array_t::new(Box::new(ArrayD::from_elem(vec![3, 4, 2], 1.0)));

            assert_eq!(data.shape().unwrap(), [3, 4, 2]);
            data.reshape(&[12, 2]).unwrap();
            assert_eq!(data.shape().unwrap(), [12, 2]);
            assert_eq!(data.as_array(), ArrayD::from_elem(vec![12, 2], 1.0));

            data.swap_axes(0, 1).unwrap();
            assert_eq!(data.shape().unwrap(), [2, 12]);
        }

        #[test]
        fn create() {
            let data = aml_array_t::new(Box::new(ArrayD::from_elem(vec![4, 2], 1.0)));

            assert_eq!(get_data_origin(data.origin().unwrap()), "rust.ndarray");
            assert_eq!(data.as_array(), ArrayD::from_elem(vec![4, 2], 1.0));

            let other = data.create(&[5, 3, 7, 12]).unwrap();
            assert_eq!(other.shape().unwrap(), [5, 3, 7, 12]);
            assert_eq!(get_data_origin(other.origin().unwrap()), "rust.ndarray");
            assert_eq!(other.as_array(), ArrayD::from_elem(vec![5, 3, 7, 12], 0.0));
        }

        #[test]
        fn move_sample() {
            let data = aml_array_t::new(Box::new(ArrayD::from_elem(vec![3, 2, 2, 4], 1.0)));

            let mut other = data.create(&[1, 2, 2, 8]).unwrap();
            assert_eq!(other.as_array(), ArrayD::from_elem(vec![1, 2, 2, 8], 0.0));

            other.move_sample(0, 2..6, &data, 1).unwrap();
            let expected = ArrayD::from_shape_vec(vec![1, 2, 2, 8], vec![
                 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
            ]).unwrap();
            assert_eq!(other.as_array(), expected);
        }
    }
}
