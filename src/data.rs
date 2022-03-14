use std::os::raw::c_void;
use std::sync::Mutex;
use std::ops::Range;

use once_cell::sync::Lazy;

use crate::{c_api::{aml_status_t, catch_unwind}, check_pointers};

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

/// `aml_array_t` manages 3D arrays the be used as data in a block/descriptor.
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
    ptr: *mut c_void,

    /// This function needs to store the "data origin" for this array in
    /// `origin`. Users of `aml_array_t` should register a single data
    /// origin with `register_data_origin`, and use it for all compatible
    /// arrays.
    origin: Option<unsafe extern fn(
        array: *const c_void,
        origin: *mut aml_data_origin_t
    ) -> aml_status_t>,

    /// Get the shape of the array managed by this `aml_array_t`
    shape: Option<unsafe extern fn(
        array: *const c_void,
        n_samples: *mut u64,
        n_components: *mut u64,
        n_features: *mut u64,
    ) -> aml_status_t>,

    /// Change the shape of the array managed by this `aml_array_t` to
    /// `(n_samples, n_components, n_features)`
    reshape: Option<unsafe extern fn(
        array: *mut c_void,
        n_samples: u64,
        n_components: u64,
        n_features: u64,
    ) -> aml_status_t>,

    /// Create a new array with the same options as the current one (data type,
    /// data location, etc.) and the requested `(n_samples, n_components,
    /// n_features)` shape; and store it in `new_array`.
    create: Option<unsafe extern fn(
        array: *const c_void,
        n_samples: u64,
        n_components: u64,
        n_features: u64,
        new_array: *mut aml_array_t,
    ) -> aml_status_t>,

    /// Set entries in this array taking data from the `other_array`. This array
    /// is guaranteed to be created by calling `aml_array_t::create` with one of
    /// the arrays in the same block or descriptor as this `array`.
    ///
    /// This function should copy data from `other_array[other_sample, :, :]` to
    /// `array[sample, :, feature_start:feature_end]`. All indexes are 0-based.
    set_from: Option<unsafe extern fn(
        array: *mut c_void,
        sample: u64,
        feature_start: u64,
        feature_end: u64,
        other_array: *const c_void,
        other_sample: u64
    ) -> aml_status_t>,

    /// Remove this array & free the associated memory. This function can be
    /// set to `NULL` is there is no memory management to do.
    pub (crate) destroy: Option<unsafe extern fn(array: *mut c_void)>,
}

impl std::fmt::Debug for aml_array_t {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut origin = None;
        let mut shape = None;
        if !self.ptr.is_null() {
            origin = Some(get_data_origin(self.origin()));
            shape = Some(self.shape());
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
            create: Some(rust_data_create),
            destroy: Some(rust_data_destroy),
            set_from: Some(rust_data_set_from),
        }
    }

    /// Create an `aml_array_t` with all fields set to null pointers.
    fn null() -> aml_array_t {
        aml_array_t {
            ptr: std::ptr::null_mut(),
            origin: None,
            set_from: None,
            shape: None,
            reshape: None,
            create: None,
            destroy: None,
        }
    }

    /// Get the origin of this array
    pub fn origin(&self) -> aml_data_origin_t {
        let function = self.origin.expect("aml_array_t.origin function is NULL");

        let mut origin = aml_data_origin_t(0);
        let status = unsafe {
            function(self.ptr, &mut origin)
        };

        assert!(status.is_success(), "aml_array_t.origin failed");

        return origin;
    }

    /// Get the shape of this array
    #[allow(clippy::cast_possible_truncation)]
    pub fn shape(&self) -> (usize, usize, usize) {
        let function = self.shape.expect("aml_array_t.shape function is NULL");

        let mut n_samples = 0;
        let mut n_components = 0;
        let mut n_features = 0;

        let status = unsafe {
            function(
                self.ptr,
                &mut n_samples,
                &mut n_components,
                &mut n_features,
            )
        };

        assert!(status.is_success(), "aml_array_t.shape failed");

        return (n_samples as usize, n_components as usize, n_features as usize);
    }

    /// Set the shape of this array to the given new `shape`
    pub fn reshape(&mut self, shape: (usize, usize, usize)) {
        let function = self.reshape.expect("aml_array_t.reshape function is NULL");

        let status = unsafe {
            function(
                self.ptr,
                shape.0 as u64,
                shape.1 as u64,
                shape.2 as u64,
            )
        };

        assert!(status.is_success(), "aml_array_t.reshape failed");
    }

    /// Create a new array with the same settings as this one and the given `shape`
    #[must_use]
    pub fn create(&self, shape: (usize, usize, usize)) -> aml_array_t {
        let function = self.create.expect("aml_array_t.create function is NULL");

        let mut data_storage = aml_array_t::null();
        let status = unsafe {
            function(
                self.ptr,
                shape.0 as u64,
                shape.1 as u64,
                shape.2 as u64,
                &mut data_storage
            )
        };

        assert!(status.is_success(), "aml_array_t.create failed");

        return data_storage;
    }

    /// Set entries in this array taking data from the `other` array. This array
    /// MUST have been created by calling `aml_array_t.create()` with one of the
    /// other arrays in the same block or descriptor.
    ///
    /// This function will copy data from `other_array[other_sample, :, :]` to
    /// `array[sample, :, feature_start:feature_end]`.
    pub fn set_from(&mut self, sample: usize, features: std::ops::Range<usize>, other: &aml_array_t, other_sample: usize) {
        let function = self.set_from.expect("aml_array_t.set_from function is NULL");

        let status = unsafe {
            function(
                self.ptr,
                sample as u64,
                features.start as u64,
                features.end as u64,
                other.ptr,
                other_sample as u64,
            )
        };

        assert!(status.is_success(), "aml_array_t.set_from failed");
    }

    /// Get the data in this `aml_array_t` as a `ndarray::Array3`. This function
    /// will panic if the data in this `aml_array_t` is not a `ndarray::Array3`.
    #[cfg(feature = "ndarray")]
    pub fn as_array(&self) -> &ndarray::Array3<f64> {
        assert_eq!(self.origin(), *NDARRAY_DATA_ORIGIN, "this array was not create by rust ndarray");

        let array = self.ptr.cast::<Box<dyn DataStorage>>();
        unsafe {
            (*array).as_any().downcast_ref().expect("invalid array type")
        }
    }

    #[cfg(feature = "ndarray")]
    pub fn as_array_mut(&mut self) -> &mut ndarray::Array3<f64> {
        assert_eq!(self.origin(), *NDARRAY_DATA_ORIGIN, "this array was not create by rust ndarray");

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

    fn create(&self, shape: (usize, usize, usize)) -> Box<dyn DataStorage>;

    fn shape(&self) -> (usize, usize, usize);
    fn reshape(&mut self, shape: (usize, usize, usize));

    fn set_from(
        &mut self,
        sample: usize,
        features: Range<usize>,
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
    n_samples: *mut u64,
    n_components: *mut u64,
    n_features: *mut u64,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(data, n_samples, n_components, n_features);
        let data = data.cast::<Box<dyn DataStorage>>();
        let shape = (*data).shape();

        *n_samples = shape.0 as u64;
        *n_components = shape.1 as u64;
        *n_features = shape.2 as u64;

        Ok(())
    })
}

/// Implementation of `aml_array_t.reshape` using `Box<dyn DataStorage>`
#[allow(clippy::cast_possible_truncation)]
unsafe extern fn rust_data_reshape(
    data: *mut c_void,
    n_samples: u64,
    n_components: u64,
    n_features: u64,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(data);
        let data = data.cast::<Box<dyn DataStorage>>();

        let shape = (
            n_samples as usize,
            n_components as usize,
            n_features as usize,
        );

        (*data).reshape(shape);

        Ok(())
    })
}

/// Implementation of `aml_array_t.create` using `Box<dyn DataStorage>`
#[allow(clippy::cast_possible_truncation)]
unsafe extern fn rust_data_create(
    data: *const c_void,
    n_samples: u64,
    n_components: u64,
    n_features: u64,
    data_storage: *mut aml_array_t,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(data, data_storage);
        let data = data.cast::<Box<dyn DataStorage>>();

        let shape = (
            n_samples as usize,
            n_components as usize,
            n_features as usize,
        );

        let new_data = (*data).create(shape);

        *data_storage = aml_array_t::new(new_data);

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

/// Implementation of `aml_array_t.set_from` using `Box<dyn DataStorage>`
#[allow(clippy::cast_possible_truncation)]
unsafe extern fn rust_data_set_from(
    data: *mut c_void,
    sample: u64,
    feature_start: u64,
    feature_end: u64,
    other: *const c_void,
    other_sample: u64
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(data, other);
        let data = data.cast::<Box<dyn DataStorage>>();
        let other = other.cast::<Box<dyn DataStorage>>();

        (*data).set_from(
            sample as usize,
            feature_start as usize .. feature_end as usize,
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
impl DataStorage for ndarray::Array3<f64> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn origin(&self) -> DataOrigin {
        return *NDARRAY_DATA_ORIGIN;
    }

    fn create(&self, shape: (usize, usize, usize)) -> Box<dyn DataStorage> {
        return Box::new(ndarray::Array3::from_elem(shape, 0.0));
    }

    fn shape(&self) -> (usize, usize, usize) {
        let shape = self.shape();
        return (shape[0], shape[1], shape[2]);
    }

    fn reshape(&mut self, shape: (usize, usize, usize)) {
        let mut array = std::mem::take(self);
        array = array.into_shape(shape).expect("invalid shape");
        std::mem::swap(self, &mut array);
    }

    fn set_from(
        &mut self,
        sample: usize,
        features: Range<usize>,
        other: &dyn DataStorage,
        sample_other: usize
    ) {
        use ndarray::s;

        let other = other.as_any().downcast_ref::<ndarray::Array3<f64>>().expect("other must be a ndarray");

        let value = other.slice(s![sample_other, .., ..]);
        self.slice_mut(s![sample, .., features]).assign(&value);
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
        shape: (usize, usize, usize),
    }

    impl TestArray {
        pub fn new(shape: (usize, usize, usize)) -> TestArray {
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

        fn create(&self, shape: (usize, usize, usize)) -> Box<dyn DataStorage> {
            Box::new(TestArray { shape: shape })
        }

        fn shape(&self) -> (usize, usize, usize) {
            self.shape
        }

        fn reshape(&mut self, shape: (usize, usize, usize)) {
            self.shape = shape;
        }

        fn set_from(
            &mut self,
            _sample: usize,
            _features: std::ops::Range<usize>,
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
        let data = aml_array_t::new(Box::new(TestArray::new((3, 4, 5))));

        let debug_format = format!("{:?}", data);
        assert_eq!(debug_format, format!(
            "aml_array_t {{ ptr: {:?}, origin: Some(\"dummy test data\"), shape: Some((3, 4, 5)) }}",
            data.ptr
        ));
    }

    #[cfg(feature = "ndarray")]
    mod ndarray {
        use ndarray::{Array3, array};

        use crate::{aml_array_t, get_data_origin};

        #[test]
        fn shape() {
            let mut data = aml_array_t::new(Box::new(Array3::from_elem((3, 4, 2), 1.0)));

            assert_eq!(data.shape(), (3, 4, 2));
            data.reshape((1, 12, 2));
            assert_eq!(data.shape(), (1, 12, 2));
            assert_eq!(data.as_array(), Array3::from_elem((1, 12, 2), 1.0));
        }

        #[test]
        fn create() {
            let data = aml_array_t::new(Box::new(Array3::from_elem((3, 4, 2), 1.0)));

            assert_eq!(get_data_origin(data.origin()), "rust.ndarray");
            assert_eq!(data.as_array(), Array3::from_elem((3, 4, 2), 1.0));

            let other = data.create((1, 12, 2));
            assert_eq!(other.shape(), (1, 12, 2));
            assert_eq!(get_data_origin(other.origin()), "rust.ndarray");
            assert_eq!(other.as_array(), Array3::from_elem((1, 12, 2), 0.0));
        }

        #[test]
        fn set_from() {
            let data = aml_array_t::new(Box::new(Array3::from_elem((3, 4, 2), 1.0)));

            let mut other = data.create((1, 4, 6));
            assert_eq!(other.as_array(), Array3::from_elem((1, 4, 6), 0.0));

            other.set_from(0, 2..4, &data, 1);
            assert_eq!(other.as_array(), array![[
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
            ]]);
        }
    }
}
