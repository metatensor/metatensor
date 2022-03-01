use std::os::raw::c_void;
use std::sync::Mutex;

use once_cell::sync::Lazy;

use crate::c_api::aml_status_t;

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

pub fn register_data_origin(origin: String) -> DataOrigin {
    let mut registered_origins = REGISTERED_DATA_ORIGIN.lock().expect("mutex got poisoned");

    for (i, registered) in registered_origins.iter().enumerate() {
        if registered == &origin {
            return aml_data_origin_t(i as u64);
        }
    }

    // could not find the origin, register a new one
    registered_origins.push(origin);

    return aml_data_origin_t((registered_origins.len() - 1) as u64);
}

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

// pub trait DataStorage: std::any::Any + RefUnwindSafe {
//     fn origin(&self) -> DataOrigin;
//     // fn data(&self) -> &[f64]; // TODO: this does not work for on-GPU data

//     fn set_from_other(
//         &mut self,
//         sample: usize,
//         features: Range<usize>,
//         other: &dyn DataStorage,
//         sample_other: usize
//     );

//     fn create(&self, shape: (usize, usize, usize)) -> Box<dyn DataStorage>;

//     // fn shape(&self) -> (usize, usize, usize);
//     fn reshape(&mut self, shape: (usize, usize, usize));
// }

#[repr(C)]
pub struct aml_data_storage_t {
    /// User-provided data should be stored here, it will be passed as the
    /// first parameter to all function pointers below.
    data: *mut c_void,

    origin: Option<unsafe extern fn(this: *const c_void, origin: *mut aml_data_origin_t) -> aml_status_t>,

    set_from_other: Option<unsafe extern fn(
        this: *const c_void,
        sample: u64,
        feature_start: u64,
        feature_end: u64,
        other: *const c_void,
        other_sample: u64
    ) -> aml_status_t>,

    reshape: Option<unsafe extern fn(
        this: *const c_void,
        n_samples: u64,
        n_symmetric: u64,
        n_features: u64,
    ) -> aml_status_t>,

    create: Option<unsafe extern fn(
        this: *const c_void,
        n_samples: u64,
        n_symmetric: u64,
        n_features: u64,
        data_storage: *mut aml_data_storage_t,
    ) -> aml_status_t>,

    destroy: Option<unsafe extern fn(this: *mut c_void)>,
}

impl Drop for aml_data_storage_t {
    fn drop(&mut self) {
        if let Some(function) = self.destroy {
            unsafe { function(self.data) }
        }
    }
}

impl aml_data_storage_t {
    // pub fn new(value: Box<dyn DataStorage>) -> aml_data_storage_t {
    //     todo!()
    // }

    fn null() -> aml_data_storage_t {
        aml_data_storage_t {
            data: std::ptr::null_mut(),
            origin: None,
            set_from_other: None,
            reshape: None,
            create: None,
            destroy: None,
        }
    }

    pub fn origin(&self) -> aml_data_origin_t {
        let function = self.origin.expect("aml_data_storage_t.origin function is NULL");

        let mut origin = aml_data_origin_t(0);
        let status = unsafe {
            function(self.data, &mut origin)
        };

        assert!(status.is_success(), "aml_data_storage_t.origin failed");

        return origin;
    }

    pub fn set_from_other(&mut self, sample: usize, features: std::ops::Range<usize>, other: &aml_data_storage_t, other_sample: usize) {
        let function = self.set_from_other.expect("aml_data_storage_t.set_from_other function is NULL");

        let status = unsafe {
            function(
                self.data,
                sample as u64,
                features.start as u64,
                features.end as u64,
                other.data,
                other_sample as u64,
            )
        };

        assert!(status.is_success(), "aml_data_storage_t.set_from_other failed");
    }

    pub fn reshape(&mut self, shape: (usize, usize, usize)) {
        let function = self.reshape.expect("aml_data_storage_t.reshape function is NULL");

        let status = unsafe {
            function(
                self.data,
                shape.0 as u64,
                shape.1 as u64,
                shape.2 as u64,
            )
        };

        assert!(status.is_success(), "aml_data_storage_t.reshape failed");
    }


    pub fn create(&self, shape: (usize, usize, usize)) -> aml_data_storage_t {
        let function = self.create.expect("aml_data_storage_t.create function is NULL");

        let mut data_storage = aml_data_storage_t::null();
        let status = unsafe {
            function(
                self.data,
                shape.0 as u64,
                shape.1 as u64,
                shape.2 as u64,
                &mut data_storage
            )
        };

        assert!(status.is_success(), "aml_data_storage_t.create failed");

        return data_storage;
    }
}
