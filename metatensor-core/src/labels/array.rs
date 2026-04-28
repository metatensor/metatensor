//! CPU-backed `mts_array_t` wrapping Labels values.

use std::os::raw::c_void;
use std::sync::{Arc, OnceLock};

use dlpk::sys::{
    DLDataType, DLDataTypeCode, DLDevice, DLManagedTensorVersioned,
    DLPackVersion, DLTensor,
};

use crate::Error;
use crate::c_api::{catch_unwind, mts_status_t};
use crate::data::{mts_array_t, mts_data_movement_t, mts_data_origin_t, register_data_origin};
use crate::labels::LabelValue;

/// Cached origin ID for CPU-backed labels arrays, registered once on first use.
static LABELS_ORIGIN: OnceLock<mts_data_origin_t> = OnceLock::new();

/// Internal struct backing the `mts_array_t` for Labels values.
struct LabelsValuesArray {
    /// Owned copy of the label values (i32 data)
    values: Vec<LabelValue>,
    /// 2D shape: [count, size]
    shape: [usize; 2],
    shape_dlpack: [i64; 2],
    strides_dlpack: [i64; 2],
}

#[allow(clippy::cast_possible_wrap)]
pub(super) fn create_array_from_vec(values: Vec<LabelValue>, count: usize, size: usize) -> mts_array_t {
    assert!(values.len() == count * size, "values length does not match count * size");

    let inner = Arc::new(LabelsValuesArray {
        values,
        shape: [count, size],
        shape_dlpack: [count as i64, size as i64],
        strides_dlpack: [size as i64, 1],
    });

    mts_array_t {
        ptr: Arc::into_raw(inner).cast_mut().cast(),
        origin: Some(labels_array_origin),
        device: Some(labels_array_device),
        dtype: Some(labels_array_dtype),
        as_dlpack: Some(labels_array_as_dlpack),
        shape: Some(labels_array_shape),
        reshape: Some(labels_array_reshape),
        swap_axes: Some(labels_array_swap_axes),
        create: Some(labels_array_create),
        copy: Some(labels_array_copy),
        destroy: Some(labels_array_destroy),
        move_data: Some(labels_array_move_data),
    }
}


/// Materialize CPU values from an `mts_array_t` via DLPack.
///
/// Calls `as_dlpack` with CPU device, reads the i32 data, and returns
/// it as `Vec<LabelValue>`.
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
pub(super) fn load_values_from_array(array: &mts_array_t, size: usize) -> Result<Vec<LabelValue>, Error> {
    let shape = array.shape()?;
    let count = shape[0];

    if count == 0 || size == 0 {
        return Ok(Vec::new());
    }

    if array.device()?.device_type == dlpk::DLDeviceType::kDLExtDev {
        return Err(Error::InvalidParameter(
            "labels values can not be loaded from dlpack's kDLExtDev device (i.e. torch's meta device)".into()
        ));
    }

    let cpu = DLDevice::cpu();
    let version = DLPackVersion::current();
    let tensor = array.as_dlpack(cpu, None, version)?;

    assert_eq!(tensor.device(), cpu, "labels array is not on CPU");

    let strides = tensor.strides().map_or([size as i64, 1], |s| [s[0], s[1]]);
    let contiguous = if count == 1 {
        // if a dimension has size 1, any stride is valid since it won't
        // be used to compute offsets. numpy takes advantage of this in
        // practise, so we have to support it
        if size == 1 {
            true
        } else {
            strides[1] == 1
        }
    } else {
        strides == [size as i64, 1]
    };
    assert_eq!(tensor.shape(), [count as i64, size as i64], "unexpected shape for labels array");

    let data_ptr: *const i32 = tensor.data_ptr()
            .map_err(|e| Error::InvalidParameter(format!("failed to cast pointer to i32 for Labels values: {}", e)))?;

    if contiguous {
        // If the data is contiguous, we can read it directly as a single slice.
        let slice = unsafe { std::slice::from_raw_parts(data_ptr, count * size) };
        return Ok(slice.to_vec());
    } else {
        // copy non-contiguous data into a contiguous Vec.
        let mut values = Vec::with_capacity(count * size);
        for i in 0..(count as i64) {
            for j in 0..(size as i64) {
                let offset = i * strides[0] + j * strides[1];
                let value = unsafe {
                    data_ptr.offset(offset as isize).read()
                };
                values.push(value);
            }
        }
        return Ok(values);
    }
}

macro_rules! check_pointers_non_null {
    ($pointer: ident) => {
        if $pointer.is_null() {
            return Err($crate::Error::InvalidParameter(
                format!(
                    "got invalid NULL pointer for {} at {}:{}",
                    stringify!($pointer), file!(), line!()
                )
            ));
        }
    };
    ($($pointer: ident),* $(,)?) => {
        $(check_pointers_non_null!($pointer);)*
    }
}


unsafe extern "C" fn labels_array_origin(
    array: *const c_void,
    origin: *mut mts_data_origin_t,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(array, origin);
        *origin = *LABELS_ORIGIN.get_or_init(|| register_data_origin("metatensor.Labels".into()));
        Ok(())
    })
}

unsafe extern "C" fn labels_array_device(
    array: *const c_void,
    device: *mut DLDevice,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(array, device);
        *device = DLDevice::cpu();
        Ok(())
    })
}

unsafe extern "C" fn labels_array_dtype(
    array: *const c_void,
    dtype: *mut DLDataType,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(array, dtype);
        // Labels values are always i32
        *dtype = DLDataType {
            code: DLDataTypeCode::kDLInt,
            bits: 32,
            lanes: 1,
        };
        Ok(())
    })
}

unsafe extern "C" fn labels_dlpack_deleter(tensor: *mut DLManagedTensorVersioned) {
    let _ = catch_unwind(|| {
        if !tensor.is_null() {
            let ctx = (*tensor).manager_ctx.cast::<LabelsValuesArray>();
            if !ctx.is_null() {
                let _ = Arc::from_raw(ctx);
            }
            let _ = Box::from_raw(tensor);
        }

        Ok(())
    });
}

#[allow(clippy::cast_possible_wrap)]
unsafe extern "C" fn labels_array_as_dlpack(
    array: *mut c_void,
    dl_managed_tensor: *mut *mut DLManagedTensorVersioned,
    device: DLDevice,
    stream: *const i64,
    max_version: DLPackVersion,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(array, dl_managed_tensor);
        let cpu = DLDevice::cpu();
        if device.device_type != cpu.device_type || device.device_id != cpu.device_id {
            return Err(Error::InvalidParameter(
                "labels values owned by metatensor-core only exist on CPU".into()
            ))
        }

        if !stream.is_null() {
            return Err(Error::InvalidParameter(
                "`stream` must be None for CPU data".into()
            ))
        }

        let current = DLPackVersion::current();
        // Accept any consumer with the same major version. The producer outputs
        // tensors compatible with version 1.0 (basic DLTensor), so there is no
        // minimum minor version requirement.
        if max_version.major != current.major {
            return Err(Error::InvalidParameter(
                format!("`max_version` too high in as_dlpack: got {}, we support {}", max_version.major, current.major)
            ))
        }

        let array = array.cast::<LabelsValuesArray>();
        Arc::increment_strong_count(array);
        let array = Arc::from_raw(array);

        // Point directly at the LabelsValuesArray's existing data (zero-copy).
        let data_ptr = if array.values.is_empty() {
            std::ptr::null_mut()
        } else {
            array.values.as_ptr() as *mut c_void
        };

        let dl_tensor = DLTensor {
            data: data_ptr,
            device: cpu,
            ndim: 2,
            dtype: DLDataType {
                code: DLDataTypeCode::kDLInt,
                bits: 32,
                lanes: 1,
            },
            shape: array.shape_dlpack.as_ptr().cast_mut(),
            strides: array.strides_dlpack.as_ptr().cast_mut(),
            byte_offset: 0,
        };

        let managed = Box::new(DLManagedTensorVersioned {
            version: current,
            manager_ctx: Arc::into_raw(array).cast_mut().cast(),
            deleter: Some(labels_dlpack_deleter),
            flags: dlpk::sys::DLPACK_FLAG_BITMASK_READ_ONLY,
            dl_tensor,
        });

        *dl_managed_tensor = Box::into_raw(managed);

        Ok(())
    })
}

unsafe extern "C" fn labels_array_shape(
    array: *const c_void,
    shape: *mut *const usize,
    shape_count: *mut usize,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(array, shape, shape_count);
        let array = &*array.cast::<LabelsValuesArray>();
        *shape = array.shape.as_ptr();
        *shape_count = 2;
        Ok(())
    })
}

unsafe extern "C" fn labels_array_copy(
    array: *const c_void,
    device: DLDevice,
    new_array: *mut mts_array_t,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(array, new_array);

        if device != DLDevice::cpu() {
            return Err(Error::InvalidParameter(
                "labels values owned by metatensor-core only exist on CPU".into()
            ))
        }

        let array = array.cast::<LabelsValuesArray>();
        Arc::increment_strong_count(array);
        let array = Arc::from_raw(array);

        *new_array = mts_array_t {
            ptr: Arc::into_raw(array).cast_mut().cast(),
            origin: Some(labels_array_origin),
            device: Some(labels_array_device),
            dtype: Some(labels_array_dtype),
            as_dlpack: Some(labels_array_as_dlpack),
            shape: Some(labels_array_shape),
            reshape: Some(labels_array_reshape),
            swap_axes: Some(labels_array_swap_axes),
            create: Some(labels_array_create),
            copy: Some(labels_array_copy),
            destroy: Some(labels_array_destroy),
            move_data: Some(labels_array_move_data),
        };

        Ok(())
    })
}

unsafe extern "C" fn labels_array_destroy(array: *mut c_void) {
    let _ = catch_unwind(|| {
        if !array.is_null() {
            let _ = Arc::from_raw(array.cast::<LabelsValuesArray>());
        }

        Ok(())
    });
}

unsafe extern "C" fn labels_array_reshape(
    array: *mut c_void,
    new_shape: *const usize,
    _new_shape_count: usize,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(array, new_shape);

        return Err(Error::InvalidParameter(
            "labels arrays do not support `reshape`".into()
        ));
    })
}

unsafe extern "C" fn labels_array_swap_axes(
    array: *mut c_void,
    _axis1: usize,
    _axis2: usize,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(array);

        return Err(Error::InvalidParameter(
            "labels arrays do not support `swap_axes`".into()
        ));
    })
}

unsafe extern "C" fn labels_array_create(
    array: *const c_void,
    shape: *const usize,
    _shape_count: usize,
    _fill_value: mts_array_t,
    new_array: *mut mts_array_t,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(array, shape, new_array);

        return Err(Error::InvalidParameter(
            "labels arrays do not support `create`".into()
        ));
    })
}

unsafe extern "C" fn labels_array_move_data(
    output: *mut c_void,
    input: *const c_void,
    movements: *const mts_data_movement_t,
    _movements_count: usize,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(output, input, movements);

        return Err(Error::InvalidParameter(
            "labels arrays do not support `move_data`".into()
        ));
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::labels::Labels;

    #[test]
    fn labels_values_array_shape() {
        let labels = Labels::from_vec(&["x", "y"], vec![1, 2, 3, 4]).unwrap();
        let values = labels.values();

        let shape = values.shape().unwrap();
        assert_eq!(shape, &[2, 2]);
    }

    #[test]
    fn labels_values_array_origin() {
        let labels = Labels::from_vec(&["a"], vec![10]).unwrap();
        let values = labels.values();

        let origin = values.origin().unwrap();
        let name = crate::data::get_data_origin(origin);
        assert_eq!(name, "metatensor.Labels");
    }

    #[test]
    fn labels_values_array_copy() {
        let labels = Labels::from_vec(&["a"], vec![10, 20]).unwrap();
        let values = labels.values();

        let cloned = values.copy(DLDevice::cpu()).unwrap();
        let shape = cloned.shape().unwrap();
        assert_eq!(shape, &[2, 1]);
    }

    #[test]
    fn empty_labels_values_array() {
        let labels = Labels::from_vec(&["a", "b"], Vec::new()).unwrap();
        let values = labels.values();

        let shape = values.shape().unwrap();
        assert_eq!(shape, &[0, 2]);
    }

    #[test]
    fn labels_values_array_device() {
        let labels = Labels::from_vec(&["x"], vec![1, 2]).unwrap();
        let values = labels.values();

        let device = values.device().unwrap();
        let cpu = DLDevice::cpu();
        assert_eq!(device.device_type, cpu.device_type);
        assert_eq!(device.device_id, cpu.device_id);
    }

    #[test]
    fn labels_values_array_as_dlpack() {
        let labels = Labels::from_vec(&["a", "b"], vec![1, 2, 3, 4]).unwrap();
        let values = labels.values();

        let cpu = DLDevice::cpu();
        let version = DLPackVersion::current();
        let tensor = values.as_dlpack(cpu, None, version).unwrap();

        assert_eq!(tensor.n_dims(), 2);
        assert_eq!(tensor.dtype().code, DLDataTypeCode::kDLInt);
        assert_eq!(tensor.dtype().bits, 32);
        assert_eq!(tensor.shape(), &[2, 2]);
    }
}
