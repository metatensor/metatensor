#![allow(warnings)]
//! Rust definition corresponding to metatensor-core C-API.
//!
//! This module is exported for advanced users of the metatensor crate, but
//! should not be needed by most.

#[cfg_attr(
    feature = "static",
    link(name = "metatensor", kind = "static", modifiers = "-whole-archive")
)]
#[cfg_attr(
    all(not(feature = "static"), not(target_os = "windows")),
    link(name = "metatensor", kind = "dylib")
)]
#[cfg_attr(
    all(not(feature = "static"), target_os = "windows"),
    link(name = "metatensor.dll", kind = "dylib")
)]
extern "C" {}

pub const MTS_SUCCESS: i32 = 0;
pub const MTS_INVALID_PARAMETER_ERROR: i32 = 1;
pub const MTS_IO_ERROR: i32 = 2;
pub const MTS_SERIALIZATION_ERROR: i32 = 3;
pub const MTS_BUFFER_SIZE_ERROR: i32 = 254;
pub const MTS_INTERNAL_ERROR: i32 = 255;
pub const MTS_NOT_IMPLEMENTED_ERROR: i32 = 5;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DLPackVersion {
    pub major: u32,
    pub minor: u32,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of DLPackVersion"][::std::mem::size_of::<DLPackVersion>() - 8usize];
    ["Alignment of DLPackVersion"][::std::mem::align_of::<DLPackVersion>() - 4usize];
    ["Offset of field: DLPackVersion::major"]
        [::std::mem::offset_of!(DLPackVersion, major) - 0usize];
    ["Offset of field: DLPackVersion::minor"]
        [::std::mem::offset_of!(DLPackVersion, minor) - 4usize];
};
pub const DLDeviceType_kDLCPU: DLDeviceType = 1;
pub const DLDeviceType_kDLCUDA: DLDeviceType = 2;
pub const DLDeviceType_kDLCUDAHost: DLDeviceType = 3;
pub const DLDeviceType_kDLOpenCL: DLDeviceType = 4;
pub const DLDeviceType_kDLVulkan: DLDeviceType = 7;
pub const DLDeviceType_kDLMetal: DLDeviceType = 8;
pub const DLDeviceType_kDLVPI: DLDeviceType = 9;
pub const DLDeviceType_kDLROCM: DLDeviceType = 10;
pub const DLDeviceType_kDLROCMHost: DLDeviceType = 11;
pub const DLDeviceType_kDLExtDev: DLDeviceType = 12;
pub const DLDeviceType_kDLCUDAManaged: DLDeviceType = 13;
pub const DLDeviceType_kDLOneAPI: DLDeviceType = 14;
pub const DLDeviceType_kDLWebGPU: DLDeviceType = 15;
pub const DLDeviceType_kDLHexagon: DLDeviceType = 16;
pub const DLDeviceType_kDLMAIA: DLDeviceType = 17;
pub const DLDeviceType_kDLTrn: DLDeviceType = 18;
pub type DLDeviceType = ::std::os::raw::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: i32,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of DLDevice"][::std::mem::size_of::<DLDevice>() - 8usize];
    ["Alignment of DLDevice"][::std::mem::align_of::<DLDevice>() - 4usize];
    ["Offset of field: DLDevice::device_type"]
        [::std::mem::offset_of!(DLDevice, device_type) - 0usize];
    ["Offset of field: DLDevice::device_id"][::std::mem::offset_of!(DLDevice, device_id) - 4usize];
};
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of DLDataType"][::std::mem::size_of::<DLDataType>() - 4usize];
    ["Alignment of DLDataType"][::std::mem::align_of::<DLDataType>() - 2usize];
    ["Offset of field: DLDataType::code"][::std::mem::offset_of!(DLDataType, code) - 0usize];
    ["Offset of field: DLDataType::bits"][::std::mem::offset_of!(DLDataType, bits) - 1usize];
    ["Offset of field: DLDataType::lanes"][::std::mem::offset_of!(DLDataType, lanes) - 2usize];
};
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DLTensor {
    pub data: *mut ::std::os::raw::c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of DLTensor"][::std::mem::size_of::<DLTensor>() - 48usize];
    ["Alignment of DLTensor"][::std::mem::align_of::<DLTensor>() - 8usize];
    ["Offset of field: DLTensor::data"][::std::mem::offset_of!(DLTensor, data) - 0usize];
    ["Offset of field: DLTensor::device"][::std::mem::offset_of!(DLTensor, device) - 8usize];
    ["Offset of field: DLTensor::ndim"][::std::mem::offset_of!(DLTensor, ndim) - 16usize];
    ["Offset of field: DLTensor::dtype"][::std::mem::offset_of!(DLTensor, dtype) - 20usize];
    ["Offset of field: DLTensor::shape"][::std::mem::offset_of!(DLTensor, shape) - 24usize];
    ["Offset of field: DLTensor::strides"][::std::mem::offset_of!(DLTensor, strides) - 32usize];
    ["Offset of field: DLTensor::byte_offset"]
        [::std::mem::offset_of!(DLTensor, byte_offset) - 40usize];
};
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DLManagedTensorVersioned {
    pub version: DLPackVersion,
    pub manager_ctx: *mut ::std::os::raw::c_void,
    pub deleter: ::std::option::Option<unsafe extern "C" fn(self_: *mut DLManagedTensorVersioned)>,
    pub flags: u64,
    pub dl_tensor: DLTensor,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of DLManagedTensorVersioned"]
        [::std::mem::size_of::<DLManagedTensorVersioned>() - 80usize];
    ["Alignment of DLManagedTensorVersioned"]
        [::std::mem::align_of::<DLManagedTensorVersioned>() - 8usize];
    ["Offset of field: DLManagedTensorVersioned::version"]
        [::std::mem::offset_of!(DLManagedTensorVersioned, version) - 0usize];
    ["Offset of field: DLManagedTensorVersioned::manager_ctx"]
        [::std::mem::offset_of!(DLManagedTensorVersioned, manager_ctx) - 8usize];
    ["Offset of field: DLManagedTensorVersioned::deleter"]
        [::std::mem::offset_of!(DLManagedTensorVersioned, deleter) - 16usize];
    ["Offset of field: DLManagedTensorVersioned::flags"]
        [::std::mem::offset_of!(DLManagedTensorVersioned, flags) - 24usize];
    ["Offset of field: DLManagedTensorVersioned::dl_tensor"]
        [::std::mem::offset_of!(DLManagedTensorVersioned, dl_tensor) - 32usize];
};
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mts_block_t {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mts_tensormap_t {
    _unused: [u8; 0],
}
pub type mts_status_t = i32;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mts_labels_t {
    pub internal_ptr_: *mut ::std::os::raw::c_void,
    pub names: *const *const ::std::os::raw::c_char,
    pub values: *const i32,
    pub size: usize,
    pub count: usize,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of mts_labels_t"][::std::mem::size_of::<mts_labels_t>() - 40usize];
    ["Alignment of mts_labels_t"][::std::mem::align_of::<mts_labels_t>() - 8usize];
    ["Offset of field: mts_labels_t::internal_ptr_"]
        [::std::mem::offset_of!(mts_labels_t, internal_ptr_) - 0usize];
    ["Offset of field: mts_labels_t::names"][::std::mem::offset_of!(mts_labels_t, names) - 8usize];
    ["Offset of field: mts_labels_t::values"]
        [::std::mem::offset_of!(mts_labels_t, values) - 16usize];
    ["Offset of field: mts_labels_t::size"][::std::mem::offset_of!(mts_labels_t, size) - 24usize];
    ["Offset of field: mts_labels_t::count"][::std::mem::offset_of!(mts_labels_t, count) - 32usize];
};
pub type mts_data_origin_t = u64;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mts_sample_mapping_t {
    pub input: usize,
    pub output: usize,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of mts_sample_mapping_t"][::std::mem::size_of::<mts_sample_mapping_t>() - 16usize];
    ["Alignment of mts_sample_mapping_t"][::std::mem::align_of::<mts_sample_mapping_t>() - 8usize];
    ["Offset of field: mts_sample_mapping_t::input"]
        [::std::mem::offset_of!(mts_sample_mapping_t, input) - 0usize];
    ["Offset of field: mts_sample_mapping_t::output"]
        [::std::mem::offset_of!(mts_sample_mapping_t, output) - 8usize];
};
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mts_array_t {
    pub ptr: *mut ::std::os::raw::c_void,
    pub origin: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            origin: *mut mts_data_origin_t,
        ) -> mts_status_t,
    >,
    pub data: ::std::option::Option<
        unsafe extern "C" fn(
            array: *mut ::std::os::raw::c_void,
            data: *mut *mut f64,
        ) -> mts_status_t,
    >,
    pub shape: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            shape: *mut *const usize,
            shape_count: *mut usize,
        ) -> mts_status_t,
    >,
    pub reshape: ::std::option::Option<
        unsafe extern "C" fn(
            array: *mut ::std::os::raw::c_void,
            shape: *const usize,
            shape_count: usize,
        ) -> mts_status_t,
    >,
    pub swap_axes: ::std::option::Option<
        unsafe extern "C" fn(
            array: *mut ::std::os::raw::c_void,
            axis_1: usize,
            axis_2: usize,
        ) -> mts_status_t,
    >,
    pub create: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            shape: *const usize,
            shape_count: usize,
            new_array: *mut mts_array_t,
        ) -> mts_status_t,
    >,
    pub copy: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            new_array: *mut mts_array_t,
        ) -> mts_status_t,
    >,
    pub destroy: ::std::option::Option<unsafe extern "C" fn(array: *mut ::std::os::raw::c_void)>,
    pub move_samples_from: ::std::option::Option<
        unsafe extern "C" fn(
            output: *mut ::std::os::raw::c_void,
            input: *const ::std::os::raw::c_void,
            samples: *const mts_sample_mapping_t,
            samples_count: usize,
            property_start: usize,
            property_end: usize,
        ) -> mts_status_t,
    >,
    pub as_dlpack: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            dl_managed_tensor: *mut *mut DLManagedTensorVersioned,
        ) -> mts_status_t,
    >,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of mts_array_t"][::std::mem::size_of::<mts_array_t>() - 88usize];
    ["Alignment of mts_array_t"][::std::mem::align_of::<mts_array_t>() - 8usize];
    ["Offset of field: mts_array_t::ptr"][::std::mem::offset_of!(mts_array_t, ptr) - 0usize];
    ["Offset of field: mts_array_t::origin"][::std::mem::offset_of!(mts_array_t, origin) - 8usize];
    ["Offset of field: mts_array_t::data"][::std::mem::offset_of!(mts_array_t, data) - 16usize];
    ["Offset of field: mts_array_t::shape"][::std::mem::offset_of!(mts_array_t, shape) - 24usize];
    ["Offset of field: mts_array_t::reshape"]
        [::std::mem::offset_of!(mts_array_t, reshape) - 32usize];
    ["Offset of field: mts_array_t::swap_axes"]
        [::std::mem::offset_of!(mts_array_t, swap_axes) - 40usize];
    ["Offset of field: mts_array_t::create"][::std::mem::offset_of!(mts_array_t, create) - 48usize];
    ["Offset of field: mts_array_t::copy"][::std::mem::offset_of!(mts_array_t, copy) - 56usize];
    ["Offset of field: mts_array_t::destroy"]
        [::std::mem::offset_of!(mts_array_t, destroy) - 64usize];
    ["Offset of field: mts_array_t::move_samples_from"]
        [::std::mem::offset_of!(mts_array_t, move_samples_from) - 72usize];
    ["Offset of field: mts_array_t::as_dlpack"]
        [::std::mem::offset_of!(mts_array_t, as_dlpack) - 80usize];
};
pub type mts_realloc_buffer_t = ::std::option::Option<
    unsafe extern "C" fn(
        user_data: *mut ::std::os::raw::c_void,
        ptr: *mut u8,
        new_size: usize,
    ) -> *mut u8,
>;
pub type mts_create_array_callback_t = ::std::option::Option<
    unsafe extern "C" fn(
        shape: *const usize,
        shape_count: usize,
        array: *mut mts_array_t,
    ) -> mts_status_t,
>;
unsafe extern "C" {
    pub fn mts_disable_panic_printing();
    pub fn mts_version() -> *const ::std::os::raw::c_char;
    pub fn mts_last_error() -> *const ::std::os::raw::c_char;
    #[must_use]
    pub fn mts_labels_position(
        labels: mts_labels_t,
        values: *const i32,
        values_count: usize,
        result: *mut i64,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_create(labels: *mut mts_labels_t) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_create_assume_unique(labels: *mut mts_labels_t) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_set_user_data(
        labels: mts_labels_t,
        user_data: *mut ::std::os::raw::c_void,
        user_data_delete: ::std::option::Option<
            unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void),
        >,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_user_data(
        labels: mts_labels_t,
        user_data: *mut *mut ::std::os::raw::c_void,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_clone(labels: mts_labels_t, clone: *mut mts_labels_t) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_union(
        first: mts_labels_t,
        second: mts_labels_t,
        result: *mut mts_labels_t,
        first_mapping: *mut i64,
        first_mapping_count: usize,
        second_mapping: *mut i64,
        second_mapping_count: usize,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_intersection(
        first: mts_labels_t,
        second: mts_labels_t,
        result: *mut mts_labels_t,
        first_mapping: *mut i64,
        first_mapping_count: usize,
        second_mapping: *mut i64,
        second_mapping_count: usize,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_difference(
        first: mts_labels_t,
        second: mts_labels_t,
        result: *mut mts_labels_t,
        first_mapping: *mut i64,
        first_mapping_count: usize,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_select(
        labels: mts_labels_t,
        selection: mts_labels_t,
        selected: *mut i64,
        selected_count: *mut usize,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_free(labels: *mut mts_labels_t) -> mts_status_t;
    #[must_use]
    pub fn mts_register_data_origin(
        name: *const ::std::os::raw::c_char,
        origin: *mut mts_data_origin_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_get_data_origin(
        origin: mts_data_origin_t,
        buffer: *mut ::std::os::raw::c_char,
        buffer_size: usize,
    ) -> mts_status_t;
    pub fn mts_block(
        data: mts_array_t,
        samples: mts_labels_t,
        components: *const mts_labels_t,
        components_count: usize,
        properties: mts_labels_t,
    ) -> *mut mts_block_t;
    #[must_use]
    pub fn mts_block_free(block: *mut mts_block_t) -> mts_status_t;
    pub fn mts_block_copy(block: *const mts_block_t) -> *mut mts_block_t;
    #[must_use]
    pub fn mts_block_labels(
        block: *const mts_block_t,
        axis: usize,
        labels: *mut mts_labels_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_block_gradient(
        block: *mut mts_block_t,
        parameter: *const ::std::os::raw::c_char,
        gradient: *mut *mut mts_block_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_block_data(block: *mut mts_block_t, data: *mut mts_array_t) -> mts_status_t;
    #[must_use]
    pub fn mts_block_add_gradient(
        block: *mut mts_block_t,
        parameter: *const ::std::os::raw::c_char,
        gradient: *mut mts_block_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_block_gradients_list(
        block: *const mts_block_t,
        parameters: *mut *const *const ::std::os::raw::c_char,
        parameters_count: *mut usize,
    ) -> mts_status_t;
    pub fn mts_tensormap(
        keys: mts_labels_t,
        blocks: *mut *mut mts_block_t,
        blocks_count: usize,
    ) -> *mut mts_tensormap_t;
    #[must_use]
    pub fn mts_tensormap_free(tensor: *mut mts_tensormap_t) -> mts_status_t;
    pub fn mts_tensormap_copy(tensor: *const mts_tensormap_t) -> *mut mts_tensormap_t;
    #[must_use]
    pub fn mts_tensormap_keys(
        tensor: *const mts_tensormap_t,
        keys: *mut mts_labels_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_tensormap_block_by_id(
        tensor: *mut mts_tensormap_t,
        block: *mut *mut mts_block_t,
        index: usize,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_tensormap_blocks_matching(
        tensor: *const mts_tensormap_t,
        block_indexes: *mut usize,
        count: *mut usize,
        selection: mts_labels_t,
    ) -> mts_status_t;
    pub fn mts_tensormap_keys_to_properties(
        tensor: *const mts_tensormap_t,
        keys_to_move: mts_labels_t,
        sort_samples: bool,
    ) -> *mut mts_tensormap_t;
    pub fn mts_tensormap_components_to_properties(
        tensor: *mut mts_tensormap_t,
        dimensions: *const *const ::std::os::raw::c_char,
        dimensions_count: usize,
    ) -> *mut mts_tensormap_t;
    pub fn mts_tensormap_keys_to_samples(
        tensor: *const mts_tensormap_t,
        keys_to_move: mts_labels_t,
        sort_samples: bool,
    ) -> *mut mts_tensormap_t;
    #[must_use]
    pub fn mts_labels_load(
        path: *const ::std::os::raw::c_char,
        labels: *mut mts_labels_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_load_buffer(
        buffer: *const u8,
        buffer_count: usize,
        labels: *mut mts_labels_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_save(
        path: *const ::std::os::raw::c_char,
        labels: mts_labels_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_save_buffer(
        buffer: *mut *mut u8,
        buffer_count: *mut usize,
        realloc_user_data: *mut ::std::os::raw::c_void,
        realloc: mts_realloc_buffer_t,
        labels: mts_labels_t,
    ) -> mts_status_t;
    pub fn mts_block_load(
        path: *const ::std::os::raw::c_char,
        create_array: mts_create_array_callback_t,
    ) -> *mut mts_block_t;
    pub fn mts_block_load_buffer(
        buffer: *const u8,
        buffer_count: usize,
        create_array: mts_create_array_callback_t,
    ) -> *mut mts_block_t;
    #[must_use]
    pub fn mts_block_save(
        path: *const ::std::os::raw::c_char,
        block: *const mts_block_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_block_save_buffer(
        buffer: *mut *mut u8,
        buffer_count: *mut usize,
        realloc_user_data: *mut ::std::os::raw::c_void,
        realloc: mts_realloc_buffer_t,
        block: *const mts_block_t,
    ) -> mts_status_t;
    pub fn mts_tensormap_load(
        path: *const ::std::os::raw::c_char,
        create_array: mts_create_array_callback_t,
    ) -> *mut mts_tensormap_t;
    pub fn mts_tensormap_load_buffer(
        buffer: *const u8,
        buffer_count: usize,
        create_array: mts_create_array_callback_t,
    ) -> *mut mts_tensormap_t;
    #[must_use]
    pub fn mts_tensormap_save(
        path: *const ::std::os::raw::c_char,
        tensor: *const mts_tensormap_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_tensormap_save_buffer(
        buffer: *mut *mut u8,
        buffer_count: *mut usize,
        realloc_user_data: *mut ::std::os::raw::c_void,
        realloc: mts_realloc_buffer_t,
        tensor: *const mts_tensormap_t,
    ) -> mts_status_t;
}
