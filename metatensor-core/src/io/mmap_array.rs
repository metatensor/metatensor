use std::os::raw::c_void;
use std::sync::Arc;

use memmap2::Mmap;

use dlpk::sys::{
    DLDataType, DLDevice, DLManagedTensorVersioned,
    DLPackVersion, DLTensor, DLPACK_FLAG_BITMASK_READ_ONLY,
};

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

// ============================================================================
// AlignedBytes: 8-byte aligned byte storage for dtype-generic arrays
// ============================================================================

/// Byte storage backed by `Vec<u64>` for guaranteed 8-byte alignment.
/// This covers all common numeric dtypes (f64, f32, f16, i64, i32, i16, i8, u8).
struct AlignedBytes {
    storage: Vec<u64>,
    byte_len: usize,
}

impl AlignedBytes {
    fn zeroed(byte_len: usize) -> Self {
        let n_u64 = (byte_len + 7) / 8;
        AlignedBytes {
            storage: vec![0u64; n_u64],
            byte_len,
        }
    }

    /// Copy raw bytes (potentially unaligned) into aligned storage.
    fn from_raw(src: *const u8, len: usize) -> Self {
        let n_u64 = (len + 7) / 8;
        let mut storage = vec![0u64; n_u64];
        if len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src,
                    storage.as_mut_ptr() as *mut u8,
                    len,
                );
            }
        }
        AlignedBytes { storage, byte_len: len }
    }

    fn as_ptr(&self) -> *const u8 {
        self.storage.as_ptr() as *const u8
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.storage.as_mut_ptr() as *mut u8, self.byte_len) }
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.byte_len) }
    }
}

impl Clone for AlignedBytes {
    fn clone(&self) -> Self {
        AlignedBytes {
            storage: self.storage.clone(),
            byte_len: self.byte_len,
        }
    }
}

// ============================================================================
// Helper: element size from DLDataType
// ============================================================================

fn element_size(dtype: &DLDataType) -> usize {
    (dtype.bits as usize / 8) * dtype.lanes as usize
}

// ============================================================================
// MmapArray: read-only view into memory-mapped file
// ============================================================================

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

// ============================================================================
// MmapArray callbacks
// ============================================================================

/// Context stored in the DLManagedTensorVersioned's manager_ctx.
/// When the deleter is called, this is dropped, which decrements the Arc refcount.
struct MmapDlpackContext {
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

        let elem_bytes = element_size(&array.dl_dtype);

        // Check if the mmap data is properly aligned for the element type.
        // If not, copy into an aligned buffer (Vec<u64> guarantees 8-byte alignment).
        let raw_ptr = array.data_ptr();
        let (data_ptr, aligned_data) = if array.data_len == 0 {
            (std::ptr::null_mut(), None)
        } else if elem_bytes <= 1 || raw_ptr as usize % elem_bytes == 0 {
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

        let mut ctx = Box::new(MmapDlpackContext {
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
            drop(Box::from_raw(managed.manager_ctx.cast::<MmapDlpackContext>()));
        }
    }
}

unsafe extern "C" fn mmap_array_create(
    array: *const c_void,
    shape_ptr: *const usize,
    shape_count: usize,
    new_array: *mut mts_array_t,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let source = &*array.cast::<MmapArray>();
        let shape = std::slice::from_raw_parts(shape_ptr, shape_count);
        let boxed = new_created_array(shape, source.dl_dtype);
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
        let data = AlignedBytes::from_raw(array.data_ptr(), array.data_len);
        let boxed = Box::new(MmapCreatedArray {
            tag: ArrayTag::Created,
            data: Arc::new(data),
            shape: array.shape.clone(),
            dl_dtype: array.dl_dtype,
        });
        *new_array = boxed.into_mts_array();
        Ok(())
    })
}

unsafe extern "C" fn mmap_array_destroy(array: *mut c_void) {
    if !array.is_null() {
        drop(Box::from_raw(array.cast::<MmapArray>()));
    }
}

// ============================================================================
// MmapCreatedArray: dtype-generic writable array for operations
// ============================================================================

#[repr(C)]
struct MmapCreatedArray {
    tag: ArrayTag,
    data: Arc<AlignedBytes>,
    shape: Vec<usize>,
    dl_dtype: DLDataType,
}

unsafe impl Send for MmapCreatedArray {}
unsafe impl Sync for MmapCreatedArray {}

impl MmapCreatedArray {
    fn into_mts_array(self: Box<Self>) -> mts_array_t {
        mts_array_t {
            ptr: Box::into_raw(self).cast(),
            origin: Some(created_array_origin),
            data: None,
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

/// Create a zeroed MmapCreatedArray with the given shape and dtype.
fn new_created_array(shape: &[usize], dl_dtype: DLDataType) -> Box<MmapCreatedArray> {
    let elem = element_size(&dl_dtype);
    let total_bytes = shape.iter().product::<usize>() * elem;
    Box::new(MmapCreatedArray {
        tag: ArrayTag::Created,
        data: Arc::new(AlignedBytes::zeroed(total_bytes)),
        shape: shape.to_vec(),
        dl_dtype,
    })
}

// ============================================================================
// MmapCreatedArray callbacks
// ============================================================================

unsafe extern "C" fn created_array_origin(
    _array: *const c_void,
    origin: *mut mts_data_origin_t,
) -> mts_status_t {
    *origin = register_data_origin("metatensor.MmapArray".into());
    mts_status_t(0)
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

/// DLPack context for MmapCreatedArray. Holds an Arc to the shared data
/// so the backing bytes remain valid for the lifetime of the DLPack tensor.
struct CreatedDlpackContext {
    _data: Arc<AlignedBytes>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

unsafe extern "C" fn created_array_as_dlpack(
    array: *mut c_void,
    dl_managed_tensor: *mut *mut DLManagedTensorVersioned,
    device: DLDevice,
    _stream: *const i64,
    max_version: DLPackVersion,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let array = &*array.cast::<MmapCreatedArray>();

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
                "MmapCreatedArray only supports CPU device".into()
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

        let data_ptr = if array.data.byte_len == 0 {
            std::ptr::null_mut()
        } else {
            array.data.as_ptr() as *mut c_void
        };

        let mut ctx = Box::new(CreatedDlpackContext {
            _data: Arc::clone(&array.data),
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
            deleter: Some(created_dlpack_deleter),
            flags: 0,
            dl_tensor,
        });

        *dl_managed_tensor = Box::into_raw(managed);
        Ok(())
    })
}

unsafe extern "C" fn created_dlpack_deleter(managed: *mut DLManagedTensorVersioned) {
    if !managed.is_null() {
        let managed = Box::from_raw(managed);
        if !managed.manager_ctx.is_null() {
            drop(Box::from_raw(managed.manager_ctx.cast::<CreatedDlpackContext>()));
        }
    }
}

unsafe extern "C" fn created_array_reshape(
    array: *mut c_void,
    shape_ptr: *const usize,
    shape_count: usize,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let array = &mut *array.cast::<MmapCreatedArray>();
        let shape = std::slice::from_raw_parts(shape_ptr, shape_count);

        let old_elements: usize = array.shape.iter().product();
        let new_elements: usize = shape.iter().product();
        if old_elements != new_elements {
            return Err(Error::Internal(format!(
                "cannot reshape: {} elements to {} elements",
                old_elements, new_elements
            )));
        }

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
        if axis_1 == axis_2 {
            return Ok(());
        }

        let elem = element_size(&array.dl_dtype);
        let ndim = array.shape.len();
        let total_elements: usize = array.shape.iter().product();
        if total_elements == 0 {
            array.shape.swap(axis_1, axis_2);
            return Ok(());
        }

        // Compute C-contiguous strides for original shape (in elements)
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * array.shape[i + 1];
        }

        // New shape after swap
        let mut new_shape = array.shape.clone();
        new_shape.swap(axis_1, axis_2);

        // New strides for the transposed shape
        let mut new_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }

        // Physical transpose: copy each element to its new position.
        // We snapshot the source bytes first, then write into the buffer.
        let data = Arc::make_mut(&mut array.data);
        let src_bytes = data.as_slice().to_vec();
        let dst = data.as_mut_slice();

        let mut indices = vec![0usize; ndim];
        for flat_idx in 0..total_elements {
            // Decompose flat_idx into multi-dimensional indices
            let mut remaining = flat_idx;
            for d in 0..ndim {
                indices[d] = remaining / strides[d];
                remaining %= strides[d];
            }

            // Swap the two axis indices
            indices.swap(axis_1, axis_2);

            // Compute destination flat index in the new layout
            let mut dst_flat = 0;
            for d in 0..ndim {
                dst_flat += indices[d] * new_strides[d];
            }

            let src_off = flat_idx * elem;
            let dst_off = dst_flat * elem;
            dst[dst_off..dst_off + elem]
                .copy_from_slice(&src_bytes[src_off..src_off + elem]);
        }

        array.shape = new_shape;
        Ok(())
    })
}

unsafe extern "C" fn created_array_create(
    array: *const c_void,
    shape_ptr: *const usize,
    shape_count: usize,
    new_array: *mut mts_array_t,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let source = &*array.cast::<MmapCreatedArray>();
        let shape = std::slice::from_raw_parts(shape_ptr, shape_count);
        let boxed = new_created_array(shape, source.dl_dtype);
        *new_array = boxed.into_mts_array();
        Ok(())
    })
}

unsafe extern "C" fn created_array_copy(
    array: *const c_void,
    new_array: *mut mts_array_t,
) -> mts_status_t {
    crate::c_api::catch_unwind(move || {
        let array = &*array.cast::<MmapCreatedArray>();
        let boxed = Box::new(MmapCreatedArray {
            tag: ArrayTag::Created,
            data: Arc::new((*array.data).clone()),
            shape: array.shape.clone(),
            dl_dtype: array.dl_dtype,
        });
        *new_array = boxed.into_mts_array();
        Ok(())
    })
}

// ============================================================================
// Byte-level sample moving (dtype-generic)
// ============================================================================

/// Move samples from `input_data` into `output_data` at the byte level.
///
/// Both input and output are C-contiguous byte buffers. The copy respects the
/// multi-dimensional layout: for each sample mapping, it copies the property
/// slice `[property_start..property_end]` across all middle (component)
/// dimensions.
unsafe fn move_samples_impl(
    output_data: &mut [u8],
    output_shape: &[usize],
    input_data: *const u8,
    input_shape: &[usize],
    elem: usize,
    samples: &[mts_sample_mapping_t],
    property_start: usize,
    property_end: usize,
) -> Result<(), Error> {
    let n_properties = property_end - property_start;
    let output_n_properties = *output_shape.last().unwrap_or(&0);
    let input_n_properties = *input_shape.last().unwrap_or(&0);

    let n_middle: usize = if output_shape.len() > 2 {
        output_shape[1..output_shape.len() - 1].iter().product()
    } else {
        1
    };

    let input_row_elements: usize = if input_shape.len() > 1 {
        input_shape[1..].iter().product()
    } else {
        1
    };

    let output_row_elements: usize = if output_shape.len() > 1 {
        output_shape[1..].iter().product()
    } else {
        1
    };

    for sample in samples {
        let in_row_byte = sample.input * input_row_elements * elem;
        let out_row_byte = sample.output * output_row_elements * elem;

        for mid in 0..n_middle {
            let out_off = out_row_byte + (mid * output_n_properties + property_start) * elem;
            let in_off = in_row_byte + mid * input_n_properties * elem;
            let copy_len = n_properties * elem;
            std::ptr::copy_nonoverlapping(
                input_data.add(in_off),
                output_data.as_mut_ptr().add(out_off),
                copy_len,
            );
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
        let elem = element_size(&output.dl_dtype);

        // Read the tag at offset 0 to determine the concrete input type.
        let tag = *(input as *const ArrayTag);
        let (input_data, input_shape): (*const u8, &[usize]) = match tag {
            ArrayTag::Created => {
                let inp = &*input.cast::<MmapCreatedArray>();
                (inp.data.as_ptr(), &inp.shape)
            }
            ArrayTag::Mmap => {
                let inp = &*input.cast::<MmapArray>();
                (inp.data_ptr(), &inp.shape)
            }
        };

        let data = Arc::make_mut(&mut output.data);
        move_samples_impl(
            data.as_mut_slice(), &output.shape,
            input_data, input_shape,
            elem,
            samples, property_start, property_end,
        )
    })
}

unsafe extern "C" fn created_array_destroy(array: *mut c_void) {
    if !array.is_null() {
        drop(Box::from_raw(array.cast::<MmapCreatedArray>()));
    }
}
