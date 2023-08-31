use std::ffi::CStr;

use crate::c_api::{mts_array_t, mts_data_origin_t, mts_status_t};
use crate::c_api::MTS_SUCCESS;

use crate::Error;
use crate::data::origin::get_data_origin;

use super::Array;

/// Reference to a data array in metatensor-core
///
/// The data array can come from any origin, this struct provides facilities to
/// access data that was created through the [`Array`] trait, and in particular
/// as `ndarray::ArrayD` instances.
#[derive(Debug, Clone, Copy)]
pub struct ArrayRef<'a> {
    array: mts_array_t,
    /// ArrayRef should behave like &'a mts_array_t
    marker: std::marker::PhantomData<&'a mts_array_t>,
}

impl<'a> ArrayRef<'a> {
    /// Create a new `ArrayRef` from the given raw `mts_array_t`
    ///
    /// This is a **VERY** unsafe function, creating a lifetime out of thin air.
    /// Make sure the lifetime is actually constrained by the lifetime of the
    /// owner of this `mts_array_t`.
    pub unsafe fn from_raw(array: mts_array_t) -> ArrayRef<'a> {
        ArrayRef {
            array,
            marker: std::marker::PhantomData,
        }
    }

    /// Get the underlying array as an `&dyn Any` instance.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`Array`] trait.
    #[inline]
    pub fn as_any(&self) -> &dyn std::any::Any {
        let origin = self.array.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<Box<dyn Array>>();
        unsafe {
            return (*array).as_any();
        }
    }

    /// Get a reference to the underlying array as an `&dyn Any` instance,
    /// re-using the same lifetime as the `ArrayRef`.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`Array`] trait.
    #[inline]
    pub fn to_any(self) -> &'a dyn std::any::Any {
        let origin = self.array.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<Box<dyn Array>>();
        unsafe {
            return (*array).as_any();
        }
    }

    /// Get the data in this `ArrayRef` as a `ndarray::ArrayD`. This function
    /// will panic if the data in this `mts_array_t` is not a `ndarray::ArrayD`.
    #[inline]
    pub fn as_array(&self) -> &ndarray::ArrayD<f64> {
        self.as_any().downcast_ref().expect("this is not a ndarray::ArrayD")
    }

    /// Transform this `ArrayRef` into a reference to an `ndarray::ArrayD`,
    /// keeping the lifetime of the `ArrayRef`.
    ///
    /// This function will panic if the data in this `mts_array_t` is not a
    /// `ndarray::ArrayD`.
    #[inline]
    pub fn to_array(self) -> &'a ndarray::ArrayD<f64> {
        self.to_any().downcast_ref().expect("this is not a ndarray::ArrayD")
    }

    /// Get the raw underlying `mts_array_t`
    pub fn as_raw(&self) -> &mts_array_t {
        &self.array
    }
}

/// Mutable reference to a data array in metatensor-core
///
/// The data array can come from any origin, this struct provides facilities to
/// access data that was created through the [`Array`] trait, and in particular
/// as `ndarray::ArrayD` instances.
#[derive(Debug)]
pub struct ArrayRefMut<'a> {
    array: mts_array_t,
    /// ArrayRef should behave like &'a mut mts_array_t
    marker: std::marker::PhantomData<&'a mut mts_array_t>,
}

impl<'a> ArrayRefMut<'a> {
    /// Create a new `ArrayRefMut` from the given raw `mts_array_t`
    ///
    /// This is a **VERY** unsafe function, creating a lifetime out of thin air,
    /// and allowing mutable access to the `mts_array_t`. Make sure the lifetime
    /// is actually constrained by the lifetime of the owner of this
    /// `mts_array_t`; and that the owner is mutably borrowed by this
    /// `ArrayRefMut`.
    #[inline]
    pub unsafe fn new(array: mts_array_t) -> ArrayRefMut<'a> {
        ArrayRefMut {
            array,
            marker: std::marker::PhantomData,
        }
    }

    /// Get the underlying array as an `&dyn Any` instance.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`Array`] trait.
    #[inline]
    pub fn as_any(&self) -> &dyn std::any::Any {
        let origin = self.array.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<Box<dyn Array>>();
        unsafe {
            return (*array).as_any();
        }
    }

    /// Get the underlying array as an `&dyn Any` instance,
    /// re-using the same lifetime as the `ArrayRefMut`.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`Array`] trait.
    #[inline]
    pub fn to_any(&self) -> &'a dyn std::any::Any {
        let origin = self.array.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<Box<dyn Array>>();
        unsafe {
            return (*array).as_any();
        }
    }

    /// Get the underlying array as an `&mut dyn Any` instance.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`Array`] trait.
    #[inline]
    pub fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        let origin = self.array.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<Box<dyn Array>>();
        unsafe {
            return (*array).as_any_mut();
        }
    }

    /// Get the underlying array as an `&mut dyn Any` instance, re-using the
    /// same lifetime as the `ArrayRefMut`.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`Array`] trait.
    #[inline]
    pub fn to_any_mut(self) -> &'a mut dyn std::any::Any {
        let origin = self.array.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<Box<dyn Array>>();
        unsafe {
            return (*array).as_any_mut();
        }
    }

    /// Get the data in this `ArrayRef` as a `ndarray::ArrayD`. This function
    /// will panic if the data in this `mts_array_t` is not a `ndarray::ArrayD`.
    #[inline]
    pub fn as_array(&self) -> &ndarray::ArrayD<f64> {
        self.as_any().downcast_ref().expect("this is not a ndarray::ArrayD")
    }

    /// Transform this `ArrayRefMut` into a reference to an `ndarray::ArrayD`,
    /// keeping the lifetime of the `ArrayRefMut`.
    ///
    /// This function will panic if the data in this `mts_array_t` is not a
    /// `ndarray::ArrayD`.
    #[inline]
    pub fn to_array(&self) -> &ndarray::ArrayD<f64> {
        self.to_any().downcast_ref().expect("this is not a ndarray::ArrayD")
    }

    /// Get the data in this `ArrayRef` as a mutable reference to an
    /// `ndarray::ArrayD`. This function will panic if the data in this
    /// `mts_array_t` is not a `ndarray::ArrayD`.
    #[inline]
    pub fn as_array_mut(&mut self) -> &mut ndarray::ArrayD<f64> {
        self.as_any_mut().downcast_mut().expect("this is not a ndarray::ArrayD")
    }

    /// Transform this `ArrayRefMut` into a mutable reference to an
    /// `ndarray::ArrayD`, keeping the lifetime of the `ArrayRefMut`.
    ///
    /// This function will panic if the data in this `mts_array_t` is not a
    /// `ndarray::ArrayD`.
    #[inline]
    pub fn to_array_mut(self) -> &'a mut ndarray::ArrayD<f64> {
        self.to_any_mut().downcast_mut().expect("this is not a ndarray::ArrayD")
    }

    /// Get the raw underlying `mts_array_t`
    pub fn as_raw(&self) -> &mts_array_t {
        &self.array
    }

    /// Get a mutable reference to the raw underlying `mts_array_t`
    pub fn as_raw_mut(&mut self) -> &mut mts_array_t {
        &mut self.array
    }
}

impl mts_array_t {
    /// Create an `mts_array_t` with all members set to null pointers/None
    pub fn null() -> mts_array_t {
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

    /// call `mts_array_t.origin` with a more convenient API
    pub fn origin(&self) -> Result<mts_data_origin_t, Error> {
        let function = self.origin.expect("mts_array_t.origin function is NULL");

        let mut origin = 0;
        unsafe {
            check_status_external(
                function(self.ptr, &mut origin),
                "mts_array_t.origin",
            )?;
        }

        return Ok(origin);
    }

    /// call `mts_array_t.shape` with a more convenient API
    #[allow(clippy::cast_possible_truncation)]
    pub fn shape(&self) -> Result<&[usize], Error> {
        let function = self.shape.expect("mts_array_t.shape function is NULL");

        let mut shape = std::ptr::null();
        let mut shape_count: usize = 0;

        unsafe {
            check_status_external(
                function(self.ptr, &mut shape, &mut shape_count),
                "mts_array_t.shape"
            )?;
        }

        assert!(shape_count > 0);
        let shape = unsafe {
            std::slice::from_raw_parts(shape, shape_count)
        };

        return Ok(shape);
    }

    /// call `mts_array_t.data` with a more convenient API
    pub fn data(&mut self) -> Result<&mut [f64], Error> {
        let shape = self.shape()?;
        let mut len = 1;
        for s in shape {
            len *= s;
        }

        let function = self.data.expect("mts_array_t.data function is NULL");

        let mut data_ptr = std::ptr::null_mut();
        let data = unsafe {
            check_status_external(
                function(self.ptr, &mut data_ptr),
                "mts_array_t.data"
            )?;

            if len == 0 {
                &mut []
            } else {
                assert!(!data_ptr.is_null());
                std::slice::from_raw_parts_mut(data_ptr, len)
            }
        };

        return Ok(data);
    }

    /// call `mts_array_t.reshape` with a more convenient API
    pub fn reshape(&mut self, shape: &[usize]) -> Result<(), Error> {
        let function = self.reshape.expect("mts_array_t.reshape function is NULL");

        unsafe {
            check_status_external(
                function(self.ptr, shape.as_ptr(), shape.len()),
                "mts_array_t.reshape",
            )?;
        }

        return Ok(());
    }

    /// call `mts_array_t.swap_axes` with a more convenient API
    pub fn swap_axes(&mut self, axis_1: usize, axis_2: usize) -> Result<(), Error> {
        let function = self.swap_axes.expect("mts_array_t.swap_axes function is NULL");

        unsafe {
            check_status_external(
                function(self.ptr, axis_1, axis_2),
                "mts_array_t.swap_axes",
            )?;
        }

        return Ok(());
    }

    /// call `mts_array_t.create` with a more convenient API
    pub fn create(&self, shape: &[usize]) -> Result<mts_array_t, Error> {
        let function = self.create.expect("mts_array_t.create function is NULL");

        let mut data_storage = mts_array_t {
            ptr: std::ptr::null_mut(),
            origin: None,
            data: None,
            shape: None,
            reshape: None,
            swap_axes: None,
            create: None,
            copy: None,
            destroy: None,
            move_samples_from: None
        };
        unsafe {
            check_status_external(
                function(self.ptr, shape.as_ptr(), shape.len(), &mut data_storage),
                "mts_array_t.create",
            )?;
        }

        return Ok(data_storage);
    }

    /// call `mts_array_t.move_samples_from` with a more convenient API
    pub fn move_samples_from(
        &mut self,
        input: &mts_array_t,
        samples: &[crate::c_api::mts_sample_mapping_t],
        properties: std::ops::Range<usize>,
    ) -> Result<(), Error> {
        let function = self.move_samples_from.expect("mts_array_t.move_samples_from function is NULL");

        unsafe {
            check_status_external(
                function(
                    self.ptr,
                    input.ptr,
                    samples.as_ptr(),
                    samples.len(),
                    properties.start,
                    properties.end,
                ),
                "mts_array_t.move_samples_from",
            )?;
        }

        return Ok(());
    }
}

/// Check the status code returned by arbitrary functions inside an
/// `mts_array_t`
fn check_status_external(status: mts_status_t, function: &str) -> Result<(), Error> {
    if status == MTS_SUCCESS {
        return Ok(())
    } else if status > 0 {
        let message = unsafe {
            CStr::from_ptr(crate::c_api::mts_last_error())
        };
        let message = message.to_str().expect("invalid UTF8");

        return Err(Error { code: Some(status), message: message.to_owned() });
    } else {
        return Err(Error { code: Some(status), message: format!("calling {} failed", function) });
    }
}
