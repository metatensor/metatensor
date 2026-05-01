#![allow(warnings)]
//! Rust definition corresponding to metatensor-core C-API.
//!
//! This module is exported for advanced users of the metatensor crate, but
//! should not be needed by most.

use dlpk::sys::*;

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

pub const MTS_SUCCESS: mts_status_t = 0;
pub const MTS_INVALID_PARAMETER_ERROR: mts_status_t = 1;
pub const MTS_IO_ERROR: mts_status_t = 2;
pub const MTS_SERIALIZATION_ERROR: mts_status_t = 3;
pub const MTS_BUFFER_SIZE_ERROR: mts_status_t = 4;
pub const MTS_CALLBACK_ERROR: mts_status_t = 254;
pub const MTS_INTERNAL_ERROR: mts_status_t = 255;
pub type mts_status_t = i32;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mts_block_t {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mts_labels_t {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mts_tensormap_t {
    _unused: [u8; 0],
}
pub type mts_data_origin_t = u64;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mts_data_movement_t {
    pub sample_in: usize,
    pub sample_out: usize,
    pub properties_start_in: usize,
    pub properties_start_out: usize,
    pub properties_length: usize,
}
#[test]
fn bindgen_test_layout_mts_data_movement_t() {
    const UNINIT: ::std::mem::MaybeUninit<mts_data_movement_t> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<mts_data_movement_t>(),
        40usize,
        "Size of mts_data_movement_t"
    );
    assert_eq!(
        ::std::mem::align_of::<mts_data_movement_t>(),
        8usize,
        "Alignment of mts_data_movement_t"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).sample_in) as usize - ptr as usize },
        0usize,
        "Offset of field: mts_data_movement_t::sample_in"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).sample_out) as usize - ptr as usize },
        8usize,
        "Offset of field: mts_data_movement_t::sample_out"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).properties_start_in) as usize - ptr as usize },
        16usize,
        "Offset of field: mts_data_movement_t::properties_start_in"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).properties_start_out) as usize - ptr as usize },
        24usize,
        "Offset of field: mts_data_movement_t::properties_start_out"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).properties_length) as usize - ptr as usize },
        32usize,
        "Offset of field: mts_data_movement_t::properties_length"
    );
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mts_array_t {
    pub ptr: *mut ::std::os::raw::c_void,
    pub destroy: ::std::option::Option<unsafe extern "C" fn(array: *mut ::std::os::raw::c_void)>,
    pub origin: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            origin: *mut mts_data_origin_t,
        ) -> mts_status_t,
    >,
    pub device: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            device: *mut DLDevice,
        ) -> mts_status_t,
    >,
    pub dtype: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            dtype: *mut DLDataType,
        ) -> mts_status_t,
    >,
    pub as_dlpack: ::std::option::Option<
        unsafe extern "C" fn(
            array: *mut ::std::os::raw::c_void,
            dl_managed_tensor: *mut *mut DLManagedTensorVersioned,
            device: DLDevice,
            stream: *const i64,
            max_version: DLPackVersion,
        ) -> mts_status_t,
    >,
    pub from_dlpack: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            dl_managed_tensor: *mut DLManagedTensorVersioned,
            new_array: *mut mts_array_t,
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
            fill_value: mts_array_t,
            new_array: *mut mts_array_t,
        ) -> mts_status_t,
    >,
    pub copy: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            device: DLDevice,
            new_array: *mut mts_array_t,
        ) -> mts_status_t,
    >,
    pub move_data: ::std::option::Option<
        unsafe extern "C" fn(
            output: *mut ::std::os::raw::c_void,
            input: *const ::std::os::raw::c_void,
            movements: *const mts_data_movement_t,
            movements_count: usize,
        ) -> mts_status_t,
    >,
}
#[test]
fn bindgen_test_layout_mts_array_t() {
    const UNINIT: ::std::mem::MaybeUninit<mts_array_t> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<mts_array_t>(),
        104usize,
        "Size of mts_array_t"
    );
    assert_eq!(
        ::std::mem::align_of::<mts_array_t>(),
        8usize,
        "Alignment of mts_array_t"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).ptr) as usize - ptr as usize },
        0usize,
        "Offset of field: mts_array_t::ptr"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).destroy) as usize - ptr as usize },
        8usize,
        "Offset of field: mts_array_t::destroy"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).origin) as usize - ptr as usize },
        16usize,
        "Offset of field: mts_array_t::origin"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).device) as usize - ptr as usize },
        24usize,
        "Offset of field: mts_array_t::device"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).dtype) as usize - ptr as usize },
        32usize,
        "Offset of field: mts_array_t::dtype"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).as_dlpack) as usize - ptr as usize },
        40usize,
        "Offset of field: mts_array_t::as_dlpack"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).from_dlpack) as usize - ptr as usize },
        48usize,
        "Offset of field: mts_array_t::from_dlpack"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).shape) as usize - ptr as usize },
        56usize,
        "Offset of field: mts_array_t::shape"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).reshape) as usize - ptr as usize },
        64usize,
        "Offset of field: mts_array_t::reshape"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).swap_axes) as usize - ptr as usize },
        72usize,
        "Offset of field: mts_array_t::swap_axes"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).create) as usize - ptr as usize },
        80usize,
        "Offset of field: mts_array_t::create"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).copy) as usize - ptr as usize },
        88usize,
        "Offset of field: mts_array_t::copy"
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).move_data) as usize - ptr as usize },
        96usize,
        "Offset of field: mts_array_t::move_data"
    );
}
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
        dtype: DLDataType,
        array: *mut mts_array_t,
    ) -> mts_status_t,
>;
extern "C" {
    pub fn mts_disable_panic_printing();
    pub fn mts_version() -> *const ::std::os::raw::c_char;
    #[must_use]
    pub fn mts_last_error(
        message: *mut *const ::std::os::raw::c_char,
        origin: *mut *const ::std::os::raw::c_char,
        data: *mut *mut ::std::os::raw::c_void,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_set_last_error(
        message: *const ::std::os::raw::c_char,
        origin: *const ::std::os::raw::c_char,
        data: *mut ::std::os::raw::c_void,
        data_deleter: ::std::option::Option<
            unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void),
        >,
    ) -> mts_status_t;
    pub fn mts_labels(
        dimensions: *const *const ::std::os::raw::c_char,
        dimensions_count: usize,
        array: mts_array_t,
    ) -> *const mts_labels_t;
    pub fn mts_labels_assume_unique(
        names: *const *const ::std::os::raw::c_char,
        names_count: usize,
        array: mts_array_t,
    ) -> *const mts_labels_t;
    #[must_use]
    pub fn mts_labels_dimensions(
        labels: *const mts_labels_t,
        names: *mut *const *const ::std::os::raw::c_char,
        count: *mut usize,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_values(labels: *const mts_labels_t, array: *mut mts_array_t) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_values_cpu(
        labels: *const mts_labels_t,
        values: *mut *const i32,
        count: *mut usize,
        size: *mut usize,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_position(
        labels: *const mts_labels_t,
        values: *const i32,
        values_count: usize,
        result: *mut i64,
    ) -> mts_status_t;
    pub fn mts_labels_clone(labels: *const mts_labels_t) -> *const mts_labels_t;
    #[must_use]
    pub fn mts_labels_union(
        first: *const mts_labels_t,
        second: *const mts_labels_t,
        result: *mut *const mts_labels_t,
        first_mapping: *mut i64,
        first_mapping_count: usize,
        second_mapping: *mut i64,
        second_mapping_count: usize,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_intersection(
        first: *const mts_labels_t,
        second: *const mts_labels_t,
        result: *mut *const mts_labels_t,
        first_mapping: *mut i64,
        first_mapping_count: usize,
        second_mapping: *mut i64,
        second_mapping_count: usize,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_difference(
        first: *const mts_labels_t,
        second: *const mts_labels_t,
        result: *mut *const mts_labels_t,
        first_mapping: *mut i64,
        first_mapping_count: usize,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_select(
        labels: *const mts_labels_t,
        selection: *const mts_labels_t,
        selected: *mut u64,
        selected_count: *mut usize,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_free(labels: *const mts_labels_t) -> mts_status_t;
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
        samples: *const mts_labels_t,
        components: *const *const mts_labels_t,
        components_count: usize,
        properties: *const mts_labels_t,
    ) -> *mut mts_block_t;
    #[must_use]
    pub fn mts_block_free(block: *mut mts_block_t) -> mts_status_t;
    pub fn mts_block_copy(block: *const mts_block_t) -> *mut mts_block_t;
    pub fn mts_block_labels(block: *const mts_block_t, axis: usize) -> *const mts_labels_t;
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
    #[must_use]
    pub fn mts_block_device(block: *const mts_block_t, device: *mut DLDevice) -> mts_status_t;
    #[must_use]
    pub fn mts_block_dtype(block: *const mts_block_t, dtype: *mut DLDataType) -> mts_status_t;
    pub fn mts_tensormap(
        keys: *const mts_labels_t,
        blocks: *mut *mut mts_block_t,
        blocks_count: usize,
    ) -> *mut mts_tensormap_t;
    #[must_use]
    pub fn mts_tensormap_free(tensor: *mut mts_tensormap_t) -> mts_status_t;
    pub fn mts_tensormap_copy(tensor: *const mts_tensormap_t) -> *mut mts_tensormap_t;
    pub fn mts_tensormap_keys(tensor: *const mts_tensormap_t) -> *const mts_labels_t;
    #[must_use]
    pub fn mts_tensormap_block_by_id(
        tensor: *mut mts_tensormap_t,
        block: *mut *mut mts_block_t,
        index: usize,
    ) -> mts_status_t;
    pub fn mts_tensormap_keys_to_properties(
        tensor: *const mts_tensormap_t,
        keys_to_move: *const mts_labels_t,
        fill_value: mts_array_t,
        sort_samples: bool,
    ) -> *mut mts_tensormap_t;
    pub fn mts_tensormap_components_to_properties(
        tensor: *mut mts_tensormap_t,
        dimensions: *const *const ::std::os::raw::c_char,
        dimensions_count: usize,
    ) -> *mut mts_tensormap_t;
    pub fn mts_tensormap_keys_to_samples(
        tensor: *const mts_tensormap_t,
        keys_to_move: *const mts_labels_t,
        fill_value: mts_array_t,
        sort_samples: bool,
    ) -> *mut mts_tensormap_t;
    #[must_use]
    pub fn mts_tensormap_set_info(
        tensor: *mut mts_tensormap_t,
        key: *const ::std::os::raw::c_char,
        value: *const ::std::os::raw::c_char,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_tensormap_get_info(
        tensor: *const mts_tensormap_t,
        key: *const ::std::os::raw::c_char,
        value: *mut *const ::std::os::raw::c_char,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_tensormap_info_keys(
        tensor: *const mts_tensormap_t,
        keys: *mut *const *const ::std::os::raw::c_char,
        keys_count: *mut usize,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_tensormap_device(
        tensor: *const mts_tensormap_t,
        device: *mut DLDevice,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_tensormap_dtype(
        tensor: *const mts_tensormap_t,
        dtype: *mut DLDataType,
    ) -> mts_status_t;
    pub fn mts_labels_load(path: *const ::std::os::raw::c_char) -> *const mts_labels_t;
    pub fn mts_labels_load_buffer(buffer: *const u8, buffer_count: usize) -> *const mts_labels_t;
    #[must_use]
    pub fn mts_labels_save(
        path: *const ::std::os::raw::c_char,
        labels: *const mts_labels_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_save_buffer(
        buffer: *mut *mut u8,
        buffer_count: *mut usize,
        realloc_user_data: *mut ::std::os::raw::c_void,
        realloc: mts_realloc_buffer_t,
        labels: *const mts_labels_t,
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
