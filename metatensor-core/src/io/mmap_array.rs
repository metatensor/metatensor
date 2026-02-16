use std::os::raw::c_void;
use std::sync::Arc;

use memmap2::Mmap;

use dlpk::sys::{
    DLDataType, DLDataTypeCode, DLDevice, DLManagedTensorVersioned,
    DLPackVersion, DLTensor, DLPACK_FLAG_BITMASK_READ_ONLY,
};
use dlpk::DLPackTensor;

use crate::c_api::mts_status_t;
use crate::data::{mts_array_t, mts_data_origin_t, mts_sample_mapping_t, register_data_origin};
use crate::Error;

/// Discriminant tag placed at offset 0 of every array struct (`#[repr(C)]`),
/// so `move_samples_from` can identify the concrete type of an opaque input.
#[derive(Clone, Copy, PartialEq)]
#[repr(u8)]
enum ArrayTag {
    Mmap = 1,
    Created = 2,
}

#[repr(C)]
pub(crate) struct MmapArray {
    tag: ArrayTag,
    mmap: Arc<Mmap>,
    data_offset: usize,
    data_len: usize,
    shape: Vec<usize>,
    dl_dtype: DLDataType,
}

// SAFETY: Mmap is backed by a file mapping which is readable from any thread.
// The MmapArray only provides read access to the data.
unsafe impl Send for MmapArray {}
unsafe impl Sync for MmapArray {}

impl MmapArray {
    pub fn new(
        mmap: Arc<Mmap>,
        data_offset: usize,
        data_len: usize,
        shape: Vec<usize>,
        dl_dtype: DLDataType,
    ) -> Self {
        MmapArray {
            tag: ArrayTag::Mmap,
            mmap,
            data_offset,
            data_len,
            shape,
            dl_dtype,
        }
    }

    pub fn into_mts_array(self) -> mts_array_t {
        let boxed = Box::new(self);
        mts_array_t {
            ptr: Box::into_raw(boxed).cast(),
            origin: Some(mmap_array_origin),
            data: None,
            as_dlpack: Some(mmap_array_as_dlpack),
            shape: Some(mmap_array_shape),
            reshape: None,
            swap_axes: None,
            create: Some(mmap_array_create),
            copy: Some(mmap_array_copy),
            destroy: Some(mmap_array_destroy),
            move_samples_from: None,
        }
    }

    fn data_ptr(&self) -> *const u8 {
        // SAFETY: data_offset + data_len was validated at construction time
        unsafe { self.mmap.as_ptr().add(self.data_offset) }
    }
}

/// Context stored in the DLManagedTensorVersioned's manager_ctx.
/// When the deleter is called, this is dropped, which decrements the Arc refcount.
struct DlpackContext {
    _mmap: Arc<Mmap>,
    /// Owned aligned copy of the data when the mmap offset is not properly
    /// aligned for the element type. `None` when zero-copy is possible.
    _aligned_data: Option<Vec<u64>>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

unsafe extern "C" fn mmap_array_origin(
    _array: *const c_void,
    origin: *mut mts_data_origin_t,
) -> mts_status_t {
    *origin = register_data_origin("metatensor.MmapArray".into());
    mts_status_t(0)
}

unsafe extern "C" fn mmap_array_shape(
    array: *const c_void,
    shape: *mut *const usize,
    shape_count: *mut usize,
) -> mts_status_t {
    let array = &*array.cast::<MmapArray>();
    *shape = array.shape.as_ptr();
    *shape_count = array.shape.len();
    mts_status_t(0)
}

unsafe extern "C" fn mmap_array_as_dlpack(
    array: *mut c_void,
    dl_managed_tensor: *mut *mut DLManagedTensorVersioned,
    device: DLDevice,
    _stream: *const i64,
    max_version: DLPackVersion,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let array = &*array.cast::<MmapArray>();

        let current_version = DLPackVersion::current();
        if max_version.major != current_version.major {
            return Err(Error::InvalidParameter(format!(
                "DLPack major version mismatch: metatensor uses {}, caller requested {}",
                current_version.major, max_version.major
            )));
        }

        let cpu = DLDevice::cpu();
        if device.device_type != cpu.device_type || device.device_id != cpu.device_id {
            return Err(Error::InvalidParameter(format!(
                "MmapArray only supports CPU device, requested {:?}:{}",
                device.device_type, device.device_id
            )));
        }

        let ndim = array.shape.len();
        let shape_i64: Vec<i64> = array.shape.iter().map(|&s| s as i64).collect();

        // Compute C-contiguous strides (in number of elements)
        let mut strides_i64 = vec![0i64; ndim];
        if ndim > 0 {
            strides_i64[ndim - 1] = 1;
            for i in (0..ndim - 1).rev() {
                strides_i64[i] = strides_i64[i + 1] * shape_i64[i + 1];
            }
        }

        let element_bytes = (array.dl_dtype.bits as usize / 8)
            * array.dl_dtype.lanes as usize;

        // Check if the mmap data is properly aligned for the element type.
        // If not, copy into an aligned buffer (Vec<u64> guarantees 8-byte alignment).
        let raw_ptr = array.data_ptr();
        let (data_ptr, aligned_data) = if array.data_len == 0 {
            (std::ptr::null_mut(), None)
        } else if element_bytes <= 1 || raw_ptr as usize % element_bytes == 0 {
            // Properly aligned — zero-copy from the mmap
            (raw_ptr as *mut c_void, None)
        } else {
            // Not aligned — copy to an aligned buffer
            let num_u64 = (array.data_len + 7) / 8;
            let mut buf: Vec<u64> = vec![0u64; num_u64];
            std::ptr::copy_nonoverlapping(
                raw_ptr,
                buf.as_mut_ptr() as *mut u8,
                array.data_len,
            );
            (buf.as_ptr() as *mut c_void, Some(buf))
        };

        let mut ctx = Box::new(DlpackContext {
            _mmap: Arc::clone(&array.mmap),
            _aligned_data: aligned_data,
            shape: shape_i64,
            strides: strides_i64,
        });

        let dl_tensor = DLTensor {
            data: data_ptr,
            device: cpu,
            ndim: ndim as i32,
            dtype: array.dl_dtype,
            shape: ctx.shape.as_mut_ptr(),
            strides: ctx.strides.as_mut_ptr(),
            byte_offset: 0,
        };

        let managed = Box::new(DLManagedTensorVersioned {
            version: current_version,
            manager_ctx: Box::into_raw(ctx).cast(),
            deleter: Some(mmap_dlpack_deleter),
            flags: DLPACK_FLAG_BITMASK_READ_ONLY,
            dl_tensor,
        });

        *dl_managed_tensor = Box::into_raw(managed);
        Ok(())
    })
}

unsafe extern "C" fn mmap_dlpack_deleter(managed: *mut DLManagedTensorVersioned) {
    if !managed.is_null() {
        let managed = Box::from_raw(managed);
        if !managed.manager_ctx.is_null() {
            drop(Box::from_raw(managed.manager_ctx.cast::<DlpackContext>()));
        }
    }
}

unsafe extern "C" fn mmap_array_create(
    _array: *const c_void,
    shape_ptr: *const usize,
    shape_count: usize,
    new_array: *mut mts_array_t,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let shape = std::slice::from_raw_parts(shape_ptr, shape_count);
        let array = ndarray::ArcArray::from_elem(shape, 0.0_f64);
        let boxed = Box::new(MmapCreatedArray {
            tag: ArrayTag::Created,
            array,
            shape: shape.to_vec(),
        });
        *new_array = boxed.into_mts_array();
        Ok(())
    })
}

unsafe extern "C" fn mmap_array_copy(
    array: *const c_void,
    new_array: *mut mts_array_t,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let array = &*array.cast::<MmapArray>();

        let data_slice = std::slice::from_raw_parts(
            array.data_ptr(),
            array.data_len,
        );

        let element_bytes = (array.dl_dtype.bits as usize / 8) * array.dl_dtype.lanes as usize;
        let num_elements: usize = array.shape.iter().product();

        if element_bytes == 8 && array.dl_dtype.code == DLDataTypeCode::kDLFloat {
            // f64 fast path — use copy_nonoverlapping to handle unaligned mmap data
            let mut dst = vec![0.0f64; num_elements];
            std::ptr::copy_nonoverlapping(
                data_slice.as_ptr(),
                dst.as_mut_ptr() as *mut u8,
                array.data_len,
            );
            let owned = ndarray::ArcArray::from_shape_vec(
                ndarray::IxDyn(&array.shape),
                dst,
            ).map_err(|e| Error::Internal(format!("shape mismatch in mmap_array_copy: {}", e)))?;
            let boxed = Box::new(MmapCreatedArray {
                tag: ArrayTag::Created,
                array: owned,
                shape: array.shape.clone(),
            });
            *new_array = boxed.into_mts_array();
        } else {
            // Generic path: copy raw bytes
            let owned_data = data_slice.to_vec();
            let boxed = Box::new(OwnedBytesArray {
                data: owned_data,
                shape: array.shape.clone(),
                dl_dtype: array.dl_dtype,
            });
            *new_array = boxed.into_mts_array();
        }

        Ok(())
    })
}

unsafe extern "C" fn mmap_array_destroy(array: *mut c_void) {
    if !array.is_null() {
        drop(Box::from_raw(array.cast::<MmapArray>()));
    }
}

// ============================================================================
// MmapCreatedArray: writable f64 array for operations like keys_to_properties
// ============================================================================

#[repr(C)]
struct MmapCreatedArray {
    tag: ArrayTag,
    array: ndarray::ArcArray<f64, ndarray::IxDyn>,
    shape: Vec<usize>,
}

unsafe impl Send for MmapCreatedArray {}
unsafe impl Sync for MmapCreatedArray {}

impl MmapCreatedArray {
    fn into_mts_array(self: Box<Self>) -> mts_array_t {
        mts_array_t {
            ptr: Box::into_raw(self).cast(),
            origin: Some(created_array_origin),
            data: Some(created_array_data),
            as_dlpack: Some(created_array_as_dlpack),
            shape: Some(created_array_shape),
            reshape: Some(created_array_reshape),
            swap_axes: Some(created_array_swap_axes),
            create: Some(created_array_create),
            copy: Some(created_array_copy),
            destroy: Some(created_array_destroy),
            move_samples_from: Some(created_array_move_samples_from),
        }
    }
}

unsafe extern "C" fn created_array_origin(
    _array: *const c_void,
    origin: *mut mts_data_origin_t,
) -> mts_status_t {
    *origin = register_data_origin("metatensor.MmapArray".into());
    mts_status_t(0)
}

unsafe extern "C" fn created_array_data(
    array: *mut c_void,
    data: *mut *mut f64,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let array = &mut *array.cast::<MmapCreatedArray>();
        let arr = array.array.as_slice_memory_order_mut()
            .ok_or_else(|| Error::Internal("created array is not contiguous".into()))?;
        *data = arr.as_mut_ptr();
        Ok(())
    })
}

unsafe extern "C" fn created_array_as_dlpack(
    array: *mut c_void,
    dl_managed_tensor: *mut *mut DLManagedTensorVersioned,
    _device: DLDevice,
    _stream: *const i64,
    _max_version: DLPackVersion,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let array = &*array.cast::<MmapCreatedArray>();

        let tensor: DLPackTensor = (&array.array).try_into().map_err(|e| {
            Error::InvalidParameter(format!("failed to convert to DLPack: {:?}", e))
        })?;

        *dl_managed_tensor = tensor.into_raw().as_ptr();
        Ok(())
    })
}

unsafe extern "C" fn created_array_shape(
    array: *const c_void,
    shape: *mut *const usize,
    shape_count: *mut usize,
) -> mts_status_t {
    let array = &*array.cast::<MmapCreatedArray>();
    *shape = array.shape.as_ptr();
    *shape_count = array.shape.len();
    mts_status_t(0)
}

unsafe extern "C" fn created_array_reshape(
    array: *mut c_void,
    shape_ptr: *const usize,
    shape_count: usize,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let array = &mut *array.cast::<MmapCreatedArray>();
        let shape = std::slice::from_raw_parts(shape_ptr, shape_count);
        array.array = array.array.clone().into_shape_with_order(ndarray::IxDyn(shape))
            .map_err(|e| Error::Internal(format!("reshape failed: {}", e)))?;
        array.shape = shape.to_vec();
        Ok(())
    })
}

unsafe extern "C" fn created_array_swap_axes(
    array: *mut c_void,
    axis_1: usize,
    axis_2: usize,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let array = &mut *array.cast::<MmapCreatedArray>();
        array.array.swap_axes(axis_1, axis_2);
        array.shape.swap(axis_1, axis_2);
        Ok(())
    })
}

unsafe extern "C" fn created_array_create(
    _array: *const c_void,
    shape_ptr: *const usize,
    shape_count: usize,
    new_array: *mut mts_array_t,
) -> mts_status_t {
    mmap_array_create(std::ptr::null(), shape_ptr, shape_count, new_array)
}

unsafe extern "C" fn created_array_copy(
    array: *const c_void,
    new_array: *mut mts_array_t,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let array = &*array.cast::<MmapCreatedArray>();
        let boxed = Box::new(MmapCreatedArray {
            tag: ArrayTag::Created,
            array: array.array.clone(),
            shape: array.shape.clone(),
        });
        *new_array = boxed.into_mts_array();
        Ok(())
    })
}

/// Shared logic for moving samples from an input f64 slice into the output ndarray.
unsafe fn move_samples_impl(
    output: &mut MmapCreatedArray,
    input_flat: &[f64],
    input_shape: &[usize],
    samples: &[mts_sample_mapping_t],
    property_start: usize,
    property_end: usize,
) -> Result<(), Error> {
    let n_properties = property_end - property_start;
    let output_n_properties = *output.shape.last().unwrap_or(&0);
    let input_n_properties = *input_shape.last().unwrap_or(&0);

    let n_middle: usize = if output.shape.len() > 2 {
        output.shape[1..output.shape.len() - 1].iter().product()
    } else {
        1
    };

    let input_row_len: usize = if input_shape.len() > 1 {
        input_shape[1..].iter().product()
    } else {
        1
    };

    for sample in samples {
        let in_row_start = sample.input * input_row_len;
        let in_row = &input_flat[in_row_start..in_row_start + input_row_len];

        let mut output_row = output.array.index_axis_mut(ndarray::Axis(0), sample.output);
        let output_flat = output_row.as_slice_mut().ok_or_else(|| {
            Error::Internal("output array is not contiguous".into())
        })?;

        for mid in 0..n_middle {
            let out_start = mid * output_n_properties + property_start;
            let in_start = mid * input_n_properties;
            output_flat[out_start..out_start + n_properties]
                .copy_from_slice(&in_row[in_start..in_start + n_properties]);
        }
    }
    Ok(())
}

unsafe extern "C" fn created_array_move_samples_from(
    output: *mut c_void,
    input: *const c_void,
    samples: *const mts_sample_mapping_t,
    samples_count: usize,
    property_start: usize,
    property_end: usize,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let output = &mut *output.cast::<MmapCreatedArray>();
        let samples = std::slice::from_raw_parts(samples, samples_count);

        // Read the tag at offset 0 to determine the concrete input type.
        let tag = *(input as *const ArrayTag);
        match tag {
            ArrayTag::Created => {
                let input = &*input.cast::<MmapCreatedArray>();
                let input_flat = input.array.as_slice_memory_order().ok_or_else(|| {
                    Error::Internal("input array is not contiguous".into())
                })?;
                move_samples_impl(
                    output, input_flat, &input.shape,
                    samples, property_start, property_end,
                )
            }
            ArrayTag::Mmap => {
                let input = &*input.cast::<MmapArray>();
                let num_elements: usize = input.shape.iter().product();
                // Alignment-safe copy from mmap into a temporary f64 buffer
                let mut buf = vec![0.0f64; num_elements];
                std::ptr::copy_nonoverlapping(
                    input.data_ptr(),
                    buf.as_mut_ptr() as *mut u8,
                    input.data_len,
                );
                move_samples_impl(
                    output, &buf, &input.shape,
                    samples, property_start, property_end,
                )
            }
        }
    })
}

unsafe extern "C" fn created_array_destroy(array: *mut c_void) {
    if !array.is_null() {
        drop(Box::from_raw(array.cast::<MmapCreatedArray>()));
    }
}

// ============================================================================
// OwnedBytesArray: for generic dtype copy from mmap
// ============================================================================

struct OwnedBytesArray {
    data: Vec<u8>,
    shape: Vec<usize>,
    dl_dtype: DLDataType,
}

unsafe impl Send for OwnedBytesArray {}
unsafe impl Sync for OwnedBytesArray {}

struct OwnedBytesDlpackContext {
    _data: Vec<u8>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

impl OwnedBytesArray {
    fn into_mts_array(self: Box<Self>) -> mts_array_t {
        mts_array_t {
            ptr: Box::into_raw(self).cast(),
            origin: Some(owned_bytes_origin),
            data: None,
            as_dlpack: Some(owned_bytes_as_dlpack),
            shape: Some(owned_bytes_shape),
            reshape: None,
            swap_axes: None,
            create: Some(owned_bytes_create),
            copy: Some(owned_bytes_copy),
            destroy: Some(owned_bytes_destroy),
            move_samples_from: None,
        }
    }
}

unsafe extern "C" fn owned_bytes_origin(
    _array: *const c_void,
    origin: *mut mts_data_origin_t,
) -> mts_status_t {
    *origin = register_data_origin("metatensor.MmapArray".into());
    mts_status_t(0)
}

unsafe extern "C" fn owned_bytes_shape(
    array: *const c_void,
    shape: *mut *const usize,
    shape_count: *mut usize,
) -> mts_status_t {
    let array = &*array.cast::<OwnedBytesArray>();
    *shape = array.shape.as_ptr();
    *shape_count = array.shape.len();
    mts_status_t(0)
}

unsafe extern "C" fn owned_bytes_as_dlpack(
    array: *mut c_void,
    dl_managed_tensor: *mut *mut DLManagedTensorVersioned,
    device: DLDevice,
    _stream: *const i64,
    max_version: DLPackVersion,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let array = &*array.cast::<OwnedBytesArray>();

        let current_version = DLPackVersion::current();
        if max_version.major != current_version.major {
            return Err(Error::InvalidParameter(format!(
                "DLPack major version mismatch: {} vs {}",
                current_version.major, max_version.major
            )));
        }

        let cpu = DLDevice::cpu();
        if device.device_type != cpu.device_type || device.device_id != cpu.device_id {
            return Err(Error::InvalidParameter(
                "OwnedBytesArray only supports CPU device".into()
            ));
        }

        let ndim = array.shape.len();
        let shape_i64: Vec<i64> = array.shape.iter().map(|&s| s as i64).collect();
        let mut strides_i64 = vec![0i64; ndim];
        if ndim > 0 {
            strides_i64[ndim - 1] = 1;
            for i in (0..ndim - 1).rev() {
                strides_i64[i] = strides_i64[i + 1] * shape_i64[i + 1];
            }
        }

        let data_ptr = if array.data.is_empty() {
            std::ptr::null_mut()
        } else {
            array.data.as_ptr() as *mut c_void
        };

        let mut ctx = Box::new(OwnedBytesDlpackContext {
            _data: array.data.clone(),
            shape: shape_i64,
            strides: strides_i64,
        });

        let dl_tensor = DLTensor {
            data: data_ptr,
            device: cpu,
            ndim: ndim as i32,
            dtype: array.dl_dtype,
            shape: ctx.shape.as_mut_ptr(),
            strides: ctx.strides.as_mut_ptr(),
            byte_offset: 0,
        };

        let managed = Box::new(DLManagedTensorVersioned {
            version: current_version,
            manager_ctx: Box::into_raw(ctx).cast(),
            deleter: Some(owned_bytes_dlpack_deleter),
            flags: 0,
            dl_tensor,
        });

        *dl_managed_tensor = Box::into_raw(managed);
        Ok(())
    })
}

unsafe extern "C" fn owned_bytes_dlpack_deleter(managed: *mut DLManagedTensorVersioned) {
    if !managed.is_null() {
        let managed = Box::from_raw(managed);
        if !managed.manager_ctx.is_null() {
            drop(Box::from_raw(managed.manager_ctx.cast::<OwnedBytesDlpackContext>()));
        }
    }
}

unsafe extern "C" fn owned_bytes_create(
    _array: *const c_void,
    shape_ptr: *const usize,
    shape_count: usize,
    new_array: *mut mts_array_t,
) -> mts_status_t {
    mmap_array_create(std::ptr::null(), shape_ptr, shape_count, new_array)
}

unsafe extern "C" fn owned_bytes_copy(
    array: *const c_void,
    new_array: *mut mts_array_t,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let array = &*array.cast::<OwnedBytesArray>();
        let boxed = Box::new(OwnedBytesArray {
            data: array.data.clone(),
            shape: array.shape.clone(),
            dl_dtype: array.dl_dtype,
        });
        *new_array = boxed.into_mts_array();
        Ok(())
    })
}

unsafe extern "C" fn owned_bytes_destroy(array: *mut c_void) {
    if !array.is_null() {
        drop(Box::from_raw(array.cast::<OwnedBytesArray>()));
    }
}
