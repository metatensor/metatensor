use std::sync::{Arc, RwLock};

use ndarray::ArrayD;

use dlpk::sys::DLDevice;

use crate::c_api::{mts_array_t, mts_data_origin_t, mts_data_movement_t};
use crate::Error;
use crate::errors::check_status;

use super::external::MtsArray;
use super::origin::get_data_origin;

/// Reference to a data array in metatensor-core
///
/// The data array can come from any origin, this struct provides facilities to
/// access data that was created through the [`crate::Array`] trait, and in particular
/// as `ndarray::ArrayD` instances.
#[derive(Debug, Clone, Copy)]
pub struct ArrayRef<'a> {
    array: mts_array_t,
    /// `ArrayRef` should behave like `&'a mts_array_t`
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
            array: mts_array_t {
                // remove the destructor if any, since we only have a reference
                // to the array, and it should not be dropped when passed back
                // to C through `as_raw()`.
                destroy: None,
                ..array
            },
            marker: std::marker::PhantomData,
        }
    }

    /// Get the underlying array as an `&dyn Any` instance.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`crate::Array`] trait.
    #[inline]
    pub fn as_any(&self) -> &dyn std::any::Any {
        let origin = self.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<super::array::RustArray>();
        unsafe {
            return (*array).as_any();
        }
    }

    /// Get a reference to the underlying array as an `&dyn Any` instance,
    /// re-using the same lifetime as the `ArrayRef`.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`crate::Array`] trait.
    #[inline]
    pub fn to_any(self) -> &'a dyn std::any::Any {
        let origin = self.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<super::array::RustArray>();
        unsafe {
            return (*array).as_any();
        }
    }

    /// Extract the `Arc<RwLock<ArrayD<T>>>` from this `ArrayRef`, if it
    /// contains one.
    ///
    /// This function will panic if the data in the `mts_array_t` in this
    /// `ArrayRef` is a different kind of array.
    #[inline]
    pub fn as_ndarray_lock<T>(&self) -> &Arc<RwLock<ArrayD<T>>> where T: 'static {
        self.as_any().downcast_ref().expect("this is not an Arc<RwLock<ArrayD>>")
    }

    /// Extract the `Arc<RwLock<ArrayD<T>>>` from this `ArrayRef`, if it
    /// contains one, keeping the initial lifetime of the `ArrayRef`.
    ///
    /// This function will panic if the data in the `mts_array_t` in this
    /// `ArrayRef` is a different kind of array.
    #[inline]
    pub fn to_ndarray_lock<T>(self) -> &'a Arc<RwLock<ArrayD<T>>> where T: 'static {
        self.to_any().downcast_ref().expect("this is not an Arc<RwLock<ArrayD>>")
    }

    /// Get the raw underlying `mts_array_t`
    pub fn as_raw(&self) -> &mts_array_t {
        &self.array
    }

    /// Get the origin of this array.
    ///
    /// This corresponds to `mts_array_t.origin`, but with a more convenient API.
    pub fn origin(&self) -> Result<mts_data_origin_t, Error> {
        let function = self.array.origin.expect("mts_array_t.origin function is NULL");

        let mut origin = 0;
        unsafe {
            check_status(function(self.array.ptr, &mut origin))?;
        }

        return Ok(origin);
    }

    /// Get the device of this array.
    ///
    /// This corresponds to `mts_array_t.device`, but with a more convenient API.
    pub fn device(&self) -> Result<DLDevice, Error> {
        let function = self.array.device.expect("mts_array_t.device function is NULL");

        let mut device = DLDevice::cpu();
        unsafe {
            check_status(function(self.array.ptr, &mut device))?;
        }

        return Ok(device);
    }

    /// Get the dtype of this array.
    ///
    /// This corresponds to `mts_array_t.dtype`, but with a more convenient API.
    pub fn dtype(&self) -> Result<dlpk::sys::DLDataType, Error> {
        let function = self.array.dtype.expect("mts_array_t.dtype function is NULL");

        let mut dtype = dlpk::sys::DLDataType { code: dlpk::sys::DLDataTypeCode::kDLFloat, bits: 0, lanes: 0 };
        unsafe {
            check_status(function(self.array.ptr, &mut dtype))?;
        }

        return Ok(dtype);
    }

    /// Get a [`dlpk::DLPackTensor`] from this array, if supported by the underlying data.
    ///
    /// This corresponds to `mts_array_t.as_dlpack`, but with a more convenient API.
    pub fn as_dlpack(
        &self,
        device: DLDevice,
        stream: Option<i64>,
        max_version: dlpk::sys::DLPackVersion,
    ) -> Result<dlpk::DLPackTensor, Error> {
        let function = self.array.as_dlpack.expect("mts_array_t.as_dlpack function is NULL");

        let mut tensor = std::ptr::null_mut();
        let stream_c = stream.as_ref().map_or(std::ptr::null(), |s| s as *const i64);

        unsafe {
            check_status(function(self.array.ptr, &mut tensor, device, stream_c, max_version))?;
        }

        let tensor = unsafe {
            dlpk::DLPackTensor::from_ptr(tensor)
        };

        return Ok(tensor);
    }

    /// Get the shape of this array.
    ///
    /// This corresponds to `mts_array_t.shape`, but with a more convenient API.
    pub fn shape(&self) -> Result<&[usize], Error> {
        let function = self.array.shape.expect("mts_array_t.shape function is NULL");

        let mut shape = std::ptr::null();
        let mut shape_count: usize = 0;

        unsafe {
            check_status(function(self.array.ptr, &mut shape, &mut shape_count))?;
        }

        if shape_count == 0 {
            return Ok(&[]);
        } else {
            assert!(!shape.is_null());
            let shape = unsafe {
                std::slice::from_raw_parts(shape, shape_count)
            };
            return Ok(shape);
        }
    }

    /// Create a new array with the same options as this one (dtype, device)
    /// and the given shape, filled with zeros.
    ///
    /// This corresponds to `mts_array_t.create`, but with a more convenient API.
    pub fn create(&self, shape: &[usize], fill_value: ArrayRef<'_>) -> Result<MtsArray, Error> {
        let function = self.array.create.expect("mts_array_t.create function is NULL");

        let mut new_array = mts_array_t::null();
        unsafe {
            check_status(function(
                self.array.ptr,
                shape.as_ptr(),
                shape.len(),
                *fill_value.as_raw(),
                &mut new_array
            ))?;
        }

        return Ok(MtsArray::from_raw(new_array));
    }

    /// Copy the data in this array, if supported by the underlying data.
    ///
    /// This corresponds to `mts_array_t.copy`, but with a more convenient API.
    pub fn copy(&self, device: DLDevice) -> Result<MtsArray, Error> {
        let function = self.array.copy.expect("mts_array_t.copy function is NULL");
        let mut new_array = mts_array_t::null();
        unsafe {
            check_status(function(self.array.ptr, device, &mut new_array))?;
        }

        return Ok(MtsArray::from_raw(new_array));
    }
}

/// Mutable reference to a data array in metatensor-core
///
/// The data array can come from any origin, this struct provides facilities to
/// access data that was created through the [`crate::Array`] trait, and in
/// particular as `ndarray::ArrayD` instances.
#[derive(Debug)]
pub struct ArrayRefMut<'a> {
    array: mts_array_t,
    /// `ArrayRefMut` should behave like `&'a mut mts_array_t`
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
    pub unsafe fn from_raw(array: mts_array_t) -> ArrayRefMut<'a> {
        ArrayRefMut {
            array: mts_array_t {
                // remove the destructor if any, since we only have a reference
                // to the array, and it should not be dropped when passed back
                // to C through `as_raw()`.
                destroy: None,
                ..array
            },
            marker: std::marker::PhantomData,
        }
    }

    /// Get the underlying array as an `&dyn Any` instance.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`crate::Array`] trait.
    #[inline]
    pub fn as_any(&self) -> &dyn std::any::Any {
        let origin = self.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<super::array::RustArray>();
        unsafe {
            return (*array).as_any();
        }
    }

    /// Get a reference to the underlying array as an `&dyn Any` instance,
    /// re-using the same lifetime as the `ArrayRefMut`.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`crate::Array`] trait.
    #[inline]
    pub fn to_any(self) -> &'a dyn std::any::Any {
        let origin = self.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<super::array::RustArray>();
        unsafe {
            return (*array).as_any();
        }
    }

    /// Get the underlying array as an `&mut dyn Any` instance.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`crate::Array`] trait.
    #[inline]
    pub fn to_any_mut(self) -> &'a mut dyn std::any::Any {
        let origin = self.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<super::array::RustArray>();
        unsafe {
            return (*array).as_any_mut();
        }
    }

    /// Extract the `Arc<RwLock<ArrayD<T>>>` from this `ArrayRef`, if it
    /// contains one.
    ///
    /// This function will panic if the data in the `mts_array_t` in this
    /// `ArrayRefMut` is a different kind of array.
    #[inline]
    pub fn as_ndarray_lock<T>(&self) -> &Arc<RwLock<ArrayD<T>>> where T: 'static {
        self.as_any().downcast_ref().expect("this is not an Arc<RwLock<ArrayD>>")
    }

    /// Extract the `Arc<RwLock<ArrayD<T>>>` from this `ArrayRef`, if it
    /// contains one, keeping the initial lifetime of the `ArrayRef`.
    ///
    /// This function will panic if the data in the `mts_array_t` in this
    /// `ArrayRefMut` is a different kind of array.
    #[inline]
    pub fn to_ndarray_lock<T>(self) -> &'a Arc<RwLock<ArrayD<T>>> where T: 'static {
        self.to_any().downcast_ref().expect("this is not an Arc<RwLock<ArrayD>>")
    }

    /// Get a mutable reference to the underlying array, consuming this
    /// `ArrayRefMut`.
    ///
    /// Since this array is already guaranteed to be unique through the mutable
    /// borrow, we do not need to lock the `RwLock` to get access to the
    /// `ArrayD`.
    ///
    /// This function will panic if the data in the `mts_array_t` in this
    /// `ArrayRefMut` does not contain an `Arc<RwLock<ArrayD<T>>>`, or if the
    /// `Arc` already has multiple strong references.
    #[inline]
    pub fn get_ndarray_mut<T>(self) -> &'a mut ArrayD<T> where T: 'static {
        let arc = self.to_any_mut().downcast_mut::<Arc<RwLock<ArrayD<T>>>>().expect("this is not an Arc<RwLock<ArrayD>>");
        let lock = Arc::get_mut(arc).expect("the outer Arc already has multiple owners");
        return lock.get_mut().expect("lock was poisoned");
    }

    /// Get the raw underlying `mts_array_t`
    pub fn as_raw(&self) -> &mts_array_t {
        &self.array
    }

    /// Get a mutable reference to the raw underlying `mts_array_t`
    pub fn as_raw_mut(&mut self) -> &mut mts_array_t {
        &mut self.array
    }

    /// Get the origin of this array.
    ///
    /// This corresponds to `mts_array_t.origin`, but with a more convenient API.
    pub fn origin(&self) -> Result<mts_data_origin_t, Error> {
        let function = self.array.origin.expect("mts_array_t.origin function is NULL");

        let mut origin = 0;
        unsafe {
            check_status(function(self.array.ptr, &mut origin))?;
        }

        return Ok(origin);
    }

    /// Get the device of this array.
    ///
    /// This corresponds to `mts_array_t.device`, but with a more convenient API.
    pub fn device(&self) -> Result<DLDevice, Error> {
        let function = self.array.device.expect("mts_array_t.device function is NULL");

        let mut device = DLDevice::cpu();
        unsafe {
            check_status(function(self.array.ptr, &mut device))?;
        }

        return Ok(device);
    }

    /// Get the dtype of this array.
    ///
    /// This corresponds to `mts_array_t.dtype`, but with a more convenient API.
    pub fn dtype(&self) -> Result<dlpk::sys::DLDataType, Error> {
        let function = self.array.dtype.expect("mts_array_t.dtype function is NULL");

        let mut dtype = dlpk::sys::DLDataType { code: dlpk::sys::DLDataTypeCode::kDLFloat, bits: 0, lanes: 0 };
        unsafe {
            check_status(function(self.array.ptr, &mut dtype))?;
        }

        return Ok(dtype);
    }

    /// Get a [`dlpk::DLPackTensor`] from this array, if supported by the underlying data.
    ///
    /// This corresponds to `mts_array_t.as_dlpack`, but with a more convenient API.
    pub fn as_dlpack(
        &self,
        device: DLDevice,
        stream: Option<i64>,
        max_version: dlpk::sys::DLPackVersion,
    ) -> Result<dlpk::DLPackTensor, Error> {
        let function = self.array.as_dlpack.expect("mts_array_t.as_dlpack function is NULL");

        let mut tensor = std::ptr::null_mut();
        let stream_c = stream.as_ref().map_or(std::ptr::null(), |s| s as *const i64);

        unsafe {
            check_status(function(
                self.array.ptr,
                &mut tensor,
                device,
                stream_c,
                max_version
            ))?;
        }

        let tensor = unsafe {
            dlpk::DLPackTensor::from_ptr(tensor)
        };

        return Ok(tensor);
    }

    /// Get the shape of this array.
    ///
    /// This corresponds to `mts_array_t.shape`, but with a more convenient API.
    pub fn shape(&self) -> Result<&[usize], Error> {
        let function = self.array.shape.expect("mts_array_t.shape function is NULL");

        let mut shape = std::ptr::null();
        let mut shape_count: usize = 0;

        unsafe {
            check_status(function(self.array.ptr, &mut shape, &mut shape_count))?;
        }

        if shape_count == 0 {
            return Ok(&[]);
        } else {
            assert!(!shape.is_null());
            let shape = unsafe {
                std::slice::from_raw_parts(shape, shape_count)
            };
            return Ok(shape);
        }
    }

    /// Reshape the data in this array, if supported by the underlying data.
    ///
    /// This corresponds to `mts_array_t.reshape`, but with a more convenient API.
    pub fn reshape(&mut self, shape: &[usize]) -> Result<(), Error> {
        let function = self.array.reshape.expect("mts_array_t.reshape function is NULL");

        unsafe {
            check_status(function(self.array.ptr, shape.as_ptr(), shape.len()))?;
        }

        return Ok(());
    }

    /// Swap two axes of the data in this array, if supported by the underlying data.
    ///
    /// This corresponds to `mts_array_t.swap_axes`, but with a more convenient API.
    pub fn swap_axes(&mut self, axis_1: usize, axis_2: usize) -> Result<(), Error> {
        let function = self.array.swap_axes.expect("mts_array_t.swap_axes function is NULL");

        unsafe {
            check_status(function(self.array.ptr, axis_1, axis_2))?;
        }

        return Ok(());
    }

    /// Create a new array with the same options as this one (dtype, device)
    /// and the given shape, filled with zeros.
    ///
    /// This corresponds to `mts_array_t.create`, but with a more convenient API.
    pub fn create(&self, shape: &[usize], fill_value: ArrayRef<'_>) -> Result<MtsArray, Error> {
        let function = self.array.create.expect("mts_array_t.create function is NULL");

        let mut new_array = mts_array_t::null();
        unsafe {
            check_status(function(
                self.array.ptr,
                shape.as_ptr(),
                shape.len(),
                *fill_value.as_raw(),
                &mut new_array
            ))?;
        }

        return Ok(MtsArray::from_raw(new_array));
    }

    /// Copy the data in this array, if supported by the underlying data.
    ///
    /// This corresponds to `mts_array_t.copy`, but with a more convenient API.
    pub fn copy(&self, device: DLDevice) -> Result<MtsArray, Error> {
        let function = self.array.copy.expect("mts_array_t.copy function is NULL");
        let mut new_array = mts_array_t::null();
        unsafe {
            check_status(function(self.array.ptr, device, &mut new_array))?;
        }

        return Ok(MtsArray::from_raw(new_array));
    }

    /// Move the data in this array to another array, if supported by the underlying data.
    ///
    /// This corresponds to `mts_array_t.move_data`, but with a more convenient API.
    pub fn move_data<'input>(
        &mut self,
        input: impl Into<ArrayRef<'input>>,
        moves: &[mts_data_movement_t],
    ) -> Result<(), Error> {
        let function = self.array.move_data.expect("mts_array_t.move_data function is NULL");

        let input = input.into();
        unsafe {
            check_status(function(
                self.array.ptr,
                input.as_raw().ptr,
                moves.as_ptr(),
                moves.len(),
            ))?;
        }

        return Ok(());
    }
}
