//! CPU-backed `mts_array_t` wrapping Labels values.
//!
//! Provides a read-only `mts_array_t` vtable for `Labels::values`, used to
//! back the `Labels::values_array()` accessor.

use std::os::raw::c_void;
use std::sync::OnceLock;

use dlpk::sys::{
    DLDataType, DLDataTypeCode, DLDevice, DLManagedTensorVersioned,
    DLPackVersion, DLTensor,
};

use crate::c_api::mts_status_t;
use crate::data::{mts_array_t, mts_data_origin_t, register_data_origin};
use crate::labels::LabelValue;

/// Cached origin ID for CPU-backed labels arrays, registered once on first use.
static LABELS_ORIGIN: OnceLock<mts_data_origin_t> = OnceLock::new();

/// Internal struct backing the `mts_array_t` for Labels values.
///
/// Holds an owned copy of the values to keep the data alive independently of
/// any particular Labels lifetime (since the array may outlive a temporary
/// reference).
pub(crate) struct LabelsValuesArray {
    /// Owned copy of the label values (i32 data)
    values: Vec<LabelValue>,
    /// 2D shape: [count, size]
    shape: [usize; 2],
}

impl LabelsValuesArray {
    /// Create a new `mts_array_t` by taking ownership of the given values Vec.
    ///
    /// `count` is the number of label entries, and `size` is the number
    /// of dimensions per entry.
    pub fn from_vec(values: Vec<LabelValue>, count: usize, size: usize) -> mts_array_t {
        let inner = Box::new(LabelsValuesArray {
            values,
            shape: [count, size],
        });

        mts_array_t {
            ptr: Box::into_raw(inner).cast(),
            origin: Some(labels_array_origin),
            device: Some(labels_array_device),
            dtype: Some(labels_array_dtype),
            as_dlpack: Some(labels_array_as_dlpack),
            shape: Some(labels_array_shape),
            reshape: None,
            swap_axes: None,
            create: None,
            copy: Some(labels_array_copy),
            destroy: Some(labels_array_destroy),
            move_data: None,
        }
    }

}

/// Materialize CPU values from an `mts_array_t` via DLPack.
///
/// Calls `as_dlpack` with CPU device, reads the i32 data, and returns
/// it as `Vec<LabelValue>`.
pub(crate) fn materialize_values_from_array(array: &mts_array_t, size: usize) -> Vec<LabelValue> {
    let shape = array.shape().expect("failed to get shape from labels array");
    let count = shape[0];

    if count == 0 || size == 0 {
        return Vec::new();
    }

    let cpu = DLDevice::cpu();
    let version = DLPackVersion::current();
    let tensor = array.as_dlpack(cpu, None, version)
        .expect("failed to get DLPack tensor from labels array");

    let ndim = tensor.n_dims();
    assert_eq!(ndim, 2, "labels DLPack tensor must be 2D");

    let dtype = tensor.dtype();
    assert_eq!(dtype.code, DLDataTypeCode::kDLInt, "labels must have integer dtype");
    assert_eq!(dtype.bits, 32, "labels must be 32-bit integers");

    let total = count * size;
    let data_ptr: *const i32 = tensor.data_ptr().expect("failed to cast DLPack data pointer to i32");
    let slice = unsafe { std::slice::from_raw_parts(data_ptr, total) };

    let values = slice.to_vec();

    // tensor is dropped here, calling the DLPack deleter

    values
}

unsafe extern "C" fn labels_array_origin(
    _array: *const c_void,
    origin: *mut mts_data_origin_t,
) -> mts_status_t {
    *origin = *LABELS_ORIGIN.get_or_init(|| register_data_origin("metatensor.labels".into()));
    mts_status_t(0)
}

unsafe extern "C" fn labels_array_device(
    _array: *const c_void,
    device: *mut DLDevice,
) -> mts_status_t {
    *device = DLDevice::cpu();
    mts_status_t(0)
}

unsafe extern "C" fn labels_array_dtype(
    _array: *const c_void,
    dtype: *mut DLDataType,
) -> mts_status_t {
    // Labels values are always i32
    *dtype = DLDataType {
        code: DLDataTypeCode::kDLInt,
        bits: 32,
        lanes: 1,
    };
    mts_status_t(0)
}

/// Context struct that keeps label data alive for the DLPack tensor lifetime.
struct LabelsDLPackContext {
    /// Owned i32 values (contiguous, row-major)
    values: Vec<i32>,
    /// DLPack shape array (i64)
    shape: [i64; 2],
    /// DLPack strides array (i64, row-major)
    strides: [i64; 2],
}

unsafe extern "C" fn labels_dlpack_deleter(tensor: *mut DLManagedTensorVersioned) {
    if !tensor.is_null() {
        let ctx = (*tensor).manager_ctx.cast::<LabelsDLPackContext>();
        if !ctx.is_null() {
            let _ = Box::from_raw(ctx);
        }
        let _ = Box::from_raw(tensor);
    }
}

#[allow(clippy::cast_possible_wrap)]
unsafe extern "C" fn labels_array_as_dlpack(
    array: *mut c_void,
    dl_managed_tensor: *mut *mut DLManagedTensorVersioned,
    device: DLDevice,
    stream: *const i64,
    max_version: DLPackVersion,
) -> mts_status_t {
    let cpu = DLDevice::cpu();
    if device.device_type != cpu.device_type || device.device_id != cpu.device_id {
        return mts_status_t(1);
    }

    if !stream.is_null() {
        return mts_status_t(1);
    }

    let current = DLPackVersion::current();
    if max_version.major != current.major || max_version.minor < current.minor {
        return mts_status_t(1);
    }

    let array_ref = &*(array as *const LabelsValuesArray);
    let count = array_ref.shape[0];
    let size = array_ref.shape[1];

    let values = array_ref.values.clone();

    let mut ctx = Box::new(LabelsDLPackContext {
        values,
        shape: [count as i64, size as i64],
        strides: [size as i64, 1],
    });

    let dl_tensor = DLTensor {
        data: ctx.values.as_mut_ptr().cast(),
        device: cpu,
        ndim: 2,
        dtype: DLDataType {
            code: DLDataTypeCode::kDLInt,
            bits: 32,
            lanes: 1,
        },
        shape: ctx.shape.as_mut_ptr(),
        strides: ctx.strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let managed = Box::new(DLManagedTensorVersioned {
        version: current,
        manager_ctx: Box::into_raw(ctx).cast(),
        deleter: Some(labels_dlpack_deleter),
        flags: 0,
        dl_tensor,
    });

    *dl_managed_tensor = Box::into_raw(managed);
    mts_status_t(0)
}

unsafe extern "C" fn labels_array_shape(
    array: *const c_void,
    shape: *mut *const usize,
    shape_count: *mut usize,
) -> mts_status_t {
    let array = &*array.cast::<LabelsValuesArray>();
    *shape = array.shape.as_ptr();
    *shape_count = 2;
    mts_status_t(0)
}

unsafe extern "C" fn labels_array_copy(
    array: *const c_void,
    new_array: *mut mts_array_t,
) -> mts_status_t {
    let array = &*array.cast::<LabelsValuesArray>();

    let cloned = Box::new(LabelsValuesArray {
        values: array.values.clone(),
        shape: array.shape,
    });

    *new_array = mts_array_t {
        ptr: Box::into_raw(cloned).cast(),
        origin: Some(labels_array_origin),
        device: Some(labels_array_device),
        dtype: Some(labels_array_dtype),
        as_dlpack: Some(labels_array_as_dlpack),
        shape: Some(labels_array_shape),
        reshape: None,
        swap_axes: None,
        create: None,
        copy: Some(labels_array_copy),
        destroy: Some(labels_array_destroy),
        move_data: None,
    };

    mts_status_t(0)
}

unsafe extern "C" fn labels_array_destroy(array: *mut c_void) {
    if !array.is_null() {
        let _ = Box::from_raw(array.cast::<LabelsValuesArray>());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labels::Labels;

    #[test]
    fn labels_values_array_shape() {
        let labels = Labels::new(&["x", "y"], vec![1.into(), 2.into(), 3.into(), 4.into()]).unwrap();
        let arr = labels.values_array();

        let shape = arr.shape().unwrap();
        assert_eq!(shape, &[2, 2]);
    }

    #[test]
    fn labels_values_array_origin() {
        let labels = Labels::new(&["a"], vec![10.into()]).unwrap();
        let arr = labels.values_array();

        let origin = arr.origin().unwrap();
        let name = crate::data::get_data_origin(origin);
        assert_eq!(name, "metatensor.labels");
    }

    #[test]
    fn labels_values_array_copy() {
        let labels = Labels::new(&["a"], vec![10.into(), 20.into()]).unwrap();
        let arr = labels.values_array();

        let cloned = arr.try_clone().unwrap();
        let shape = cloned.shape().unwrap();
        assert_eq!(shape, &[2, 1]);
    }

    #[test]
    fn empty_labels_values_array() {
        let labels = Labels::new(&["a", "b"], Vec::new()).unwrap();
        let arr = labels.values_array();

        let shape = arr.shape().unwrap();
        assert_eq!(shape, &[0, 2]);
    }

    #[test]
    fn labels_values_array_device() {
        let labels = Labels::new(&["x"], vec![1.into(), 2.into()]).unwrap();
        let arr = labels.values_array();

        let device = arr.device().unwrap();
        let cpu = DLDevice::cpu();
        assert_eq!(device.device_type, cpu.device_type);
        assert_eq!(device.device_id, cpu.device_id);
    }

    #[test]
    fn labels_values_array_as_dlpack() {
        let labels = Labels::new(&["a", "b"], vec![1.into(), 2.into(), 3.into(), 4.into()]).unwrap();
        let arr = labels.values_array();

        let cpu = DLDevice::cpu();
        let version = DLPackVersion::current();
        let tensor = arr.as_dlpack(cpu, None, version).unwrap();

        assert_eq!(tensor.n_dims(), 2);
        assert_eq!(tensor.dtype().code, DLDataTypeCode::kDLInt);
        assert_eq!(tensor.dtype().bits, 32);
        assert_eq!(tensor.shape(), &[2, 2]);
    }
}
