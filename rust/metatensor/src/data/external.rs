use std::ptr::NonNull;
use std::sync::{Arc, RwLock, RwLockReadGuard};

use ndarray::ArrayD;
use dlpk::sys::DLDevice;

use crate::c_api::{mts_array_t, mts_data_origin_t, mts_data_movement_t, mts_status_t};
use crate::c_api::{MTS_SUCCESS, mts_last_error};

use crate::Error;
use crate::errors::{LAST_RUST_ERROR, RUST_FUNCTION_FAILED_ERROR_CODE};

use super::{ArrayRef, ArrayRefMut};
use super::origin::get_data_origin;

/// Wrapper around `mts_array_t` that provides a more convenient API to use it
/// in Rust code, and in particular to access the underlying array as an `&dyn
/// Any` instance where possible.
pub struct MtsArray {
    array: mts_array_t
}

impl Drop for MtsArray {
    fn drop(&mut self) {
        if let Some(destroy) = self.array.destroy {
             unsafe { destroy(self.array.ptr) }
        }
    }
}

impl MtsArray {
    /// Create a new `MtsArray` from a `mts_array_t`, taking ownership of the
    /// data.
    pub fn from_raw(array: mts_array_t) -> MtsArray {
        MtsArray { array }
    }

    /// Get the underlying `mts_array_t`, transferring ownership of the data to
    /// the caller.
    pub fn into_raw(self) -> mts_array_t {
        let array = self.array;
        // since mts_array_t is Copy, we need to forget self to avoid calling
        // Drop when this function returns
        std::mem::forget(self);
        array
    }

    /// Get the underlying array as an `&dyn Any` instance.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`Array`] trait.
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

    #[inline]
    fn as_lock<T>(&self) -> &Arc<RwLock<ArrayD<T>>> where T: 'static {
        self.as_any().downcast_ref().expect("this is not an Arc<RwLock<ArrayD>>")
    }

    /// Get the data in this `ArrayRef` as a `ndarray::ArcArray`. This function
    /// will panic if the data in this `mts_array_t` is not a `ndarray::ArcArray`.
    #[inline]
    pub fn as_ndarray<T>(&self) -> RwLockReadGuard<'_, ArrayD<T>> where T: 'static {
        return self.as_lock().read().expect("lock was poisoned");
    }

    /// Get the underlying `mts_array_t`.
    pub fn as_raw(&self) -> &mts_array_t {
        &self.array
    }

    /// Get the underlying `mts_array_t` as a mutable reference.
    pub fn as_raw_mut(&mut self) -> &mut mts_array_t {
        &mut self.array
    }

    /// Get a reference to this array
    pub fn as_ref(&'_ self) -> ArrayRef<'_> {
        unsafe { ArrayRef::from_raw(self.array) }
    }

    /// Get a mutable reference to this array
    pub fn as_mut(&'_ mut self) -> ArrayRefMut<'_> {
        unsafe { ArrayRefMut::from_raw(self.array) }
    }

    /// Get the origin of this array.
    ///
    /// This corresponds to `mts_array_t.origin`, but with a more convenient API.
    pub fn origin(&self) -> Result<mts_data_origin_t, Error> {
        let function = self.array.origin.expect("mts_array_t.origin function is NULL");

        let mut origin = 0;
        unsafe {
            check_status_external(
                function(self.array.ptr, &mut origin),
                "mts_array_t.origin",
            )?;
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
            check_status_external(
                function(self.array.ptr, &mut device),
                "mts_array_t.device",
            )?;
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
            check_status_external(
                function(self.array.ptr, &mut dtype),
                "mts_array_t.dtype",
            )?;
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
            check_status_external(
                function(self.array.ptr, &mut tensor, device, stream_c, max_version),
                "mts_array_t.as_dlpack",
            )?;
        }

        let tensor = NonNull::new(tensor).expect("got a NULL DLManagedTensorVersioned from `as_dlpack`");
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
            check_status_external(
                function(self.array.ptr, &mut shape, &mut shape_count),
                "mts_array_t.shape"
            )?;
        }

        assert!(shape_count > 0);
        let shape = unsafe {
            std::slice::from_raw_parts(shape, shape_count)
        };

        return Ok(shape);
    }

    /// Reshape the data in this array, if supported by the underlying data.
    ///
    /// This corresponds to `mts_array_t.reshape`, but with a more convenient API.
    pub fn reshape(&mut self, shape: &[usize]) -> Result<(), Error> {
        let function = self.array.reshape.expect("mts_array_t.reshape function is NULL");

        unsafe {
            check_status_external(
                function(self.array.ptr, shape.as_ptr(), shape.len()),
                "mts_array_t.reshape",
            )?;
        }

        return Ok(());
    }

    /// Swap two axes of the data in this array, if supported by the underlying data.
    ///
    /// This corresponds to `mts_array_t.swap_axes`, but with a more convenient API.
    pub fn swap_axes(&mut self, axis_1: usize, axis_2: usize) -> Result<(), Error> {
        let function = self.array.swap_axes.expect("mts_array_t.swap_axes function is NULL");

        unsafe {
            check_status_external(
                function(self.array.ptr, axis_1, axis_2),
                "mts_array_t.swap_axes",
            )?;
        }

        return Ok(());
    }

    /// Create a new array with the same data as this array, but with a different shape.
    ///
    /// This corresponds to `mts_array_t.create`, but with a more convenient API.
    pub fn create(&self, shape: &[usize], fill_value: ArrayRef<'_>) -> Result<MtsArray, Error> {
        let function = self.array.create.expect("mts_array_t.create function is NULL");

        let mut new_array = mts_array_t::null();
        unsafe {
            check_status_external(
                function(
                    self.array.ptr,
                    shape.as_ptr(),
                    shape.len(),
                    *fill_value.as_raw(),
                    &mut new_array
                ),
                "mts_array_t.create",
            )?;
        }

        return Ok(MtsArray::from_raw(new_array));
    }

    /// Copy the data in this array, if supported by the underlying data.
    ///
    /// This corresponds to `mts_array_t.copy`, but with a more convenient API.
    pub fn copy(&self) -> Result<mts_array_t, Error> {
        let function = self.array.copy.expect("mts_array_t.copy function is NULL");
        let mut new_array = mts_array_t::null();
        unsafe {
            check_status_external(
                function(self.array.ptr, &mut new_array),
                "mts_array_t.copy",
            )?;
        }

        return Ok(new_array);
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
            check_status_external(
                function(self.array.ptr, input.as_raw().ptr, moves.as_ptr(), moves.len()),
                "mts_array_t.move_data",
            )?;
        }

        return Ok(());
    }
}

impl<'a> From<&'a MtsArray> for ArrayRef<'a> {
    fn from(array: &'a MtsArray) -> ArrayRef<'a> {
        array.as_ref()
    }
}

impl<'a> From<&'a mut MtsArray> for ArrayRefMut<'a> {
    fn from(array: &'a mut MtsArray) -> ArrayRefMut<'a> {
        array.as_mut()
    }
}

/// Check the status code returned by arbitrary functions inside an
/// `mts_array_t`
pub(super) fn check_status_external(status: mts_status_t, function: &str) -> Result<(), Error> {
    if status == MTS_SUCCESS {
        return Ok(());
    } else if status > 0 {
        let message = unsafe {
            std::ffi::CStr::from_ptr(mts_last_error())
        };
        let message = message.to_str().expect("invalid UTF8");

        return Err(Error { code: Some(status), message: message.to_owned() });
    } else if status == RUST_FUNCTION_FAILED_ERROR_CODE {
        return Err(LAST_RUST_ERROR.with(|e| e.borrow().clone()));
    } else {
        return Err(Error { code: Some(status), message: format!("calling {} failed", function) });
    }
}
