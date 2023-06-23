#![allow(warnings)]
//! Rust definition corresponding to equistore-core C-API.
//!
//! This module is exported for advanced users of the equistore crate, but
//! should not be needed by most.

#[cfg_attr(feature="static", link(name="equistore", kind = "static", modifiers = "-whole-archive"))]
#[cfg_attr(not(feature="static"), link(name="equistore", kind = "dylib"))]
extern "C" {}

pub const EQS_SUCCESS: i32 = 0;
pub const EQS_INVALID_PARAMETER_ERROR: i32 = 1;
pub const EQS_IO_ERROR: i32 = 2;
pub const EQS_SERIALIZATION_ERROR: i32 = 3;
pub const EQS_BUFFER_SIZE_ERROR: i32 = 254;
pub const EQS_INTERNAL_ERROR: i32 = 255;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct eqs_block_t {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct eqs_tensormap_t {
    _unused: [u8; 0],
}
pub type eqs_status_t = i32;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct eqs_labels_t {
    pub internal_ptr_: *mut ::std::os::raw::c_void,
    pub names: *const *const ::std::os::raw::c_char,
    pub values: *const i32,
    pub size: usize,
    pub count: usize,
}
#[test]
fn bindgen_test_layout_eqs_labels_t() {
    const UNINIT: ::std::mem::MaybeUninit<eqs_labels_t> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<eqs_labels_t>(),
        40usize,
        concat!("Size of: ", stringify!(eqs_labels_t))
    );
    assert_eq!(
        ::std::mem::align_of::<eqs_labels_t>(),
        8usize,
        concat!("Alignment of ", stringify!(eqs_labels_t))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).internal_ptr_) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_labels_t),
            "::",
            stringify!(internal_ptr_)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).names) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_labels_t),
            "::",
            stringify!(names)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).values) as usize - ptr as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_labels_t),
            "::",
            stringify!(values)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).size) as usize - ptr as usize },
        24usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_labels_t),
            "::",
            stringify!(size)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).count) as usize - ptr as usize },
        32usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_labels_t),
            "::",
            stringify!(count)
        )
    );
}
pub type eqs_data_origin_t = u64;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct eqs_sample_mapping_t {
    pub input: usize,
    pub output: usize,
}
#[test]
fn bindgen_test_layout_eqs_sample_mapping_t() {
    const UNINIT: ::std::mem::MaybeUninit<eqs_sample_mapping_t> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<eqs_sample_mapping_t>(),
        16usize,
        concat!("Size of: ", stringify!(eqs_sample_mapping_t))
    );
    assert_eq!(
        ::std::mem::align_of::<eqs_sample_mapping_t>(),
        8usize,
        concat!("Alignment of ", stringify!(eqs_sample_mapping_t))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).input) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_sample_mapping_t),
            "::",
            stringify!(input)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).output) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_sample_mapping_t),
            "::",
            stringify!(output)
        )
    );
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct eqs_array_t {
    pub ptr: *mut ::std::os::raw::c_void,
    pub origin: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            origin: *mut eqs_data_origin_t,
        ) -> eqs_status_t,
    >,
    pub data: ::std::option::Option<
        unsafe extern "C" fn(
            array: *mut ::std::os::raw::c_void,
            data: *mut *mut f64,
        ) -> eqs_status_t,
    >,
    pub shape: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            shape: *mut *const usize,
            shape_count: *mut usize,
        ) -> eqs_status_t,
    >,
    pub reshape: ::std::option::Option<
        unsafe extern "C" fn(
            array: *mut ::std::os::raw::c_void,
            shape: *const usize,
            shape_count: usize,
        ) -> eqs_status_t,
    >,
    pub swap_axes: ::std::option::Option<
        unsafe extern "C" fn(
            array: *mut ::std::os::raw::c_void,
            axis_1: usize,
            axis_2: usize,
        ) -> eqs_status_t,
    >,
    pub create: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            shape: *const usize,
            shape_count: usize,
            new_array: *mut eqs_array_t,
        ) -> eqs_status_t,
    >,
    pub copy: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            new_array: *mut eqs_array_t,
        ) -> eqs_status_t,
    >,
    pub destroy: ::std::option::Option<unsafe extern "C" fn(array: *mut ::std::os::raw::c_void)>,
    pub move_samples_from: ::std::option::Option<
        unsafe extern "C" fn(
            output: *mut ::std::os::raw::c_void,
            input: *const ::std::os::raw::c_void,
            samples: *const eqs_sample_mapping_t,
            samples_count: usize,
            property_start: usize,
            property_end: usize,
        ) -> eqs_status_t,
    >,
}
#[test]
fn bindgen_test_layout_eqs_array_t() {
    const UNINIT: ::std::mem::MaybeUninit<eqs_array_t> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<eqs_array_t>(),
        80usize,
        concat!("Size of: ", stringify!(eqs_array_t))
    );
    assert_eq!(
        ::std::mem::align_of::<eqs_array_t>(),
        8usize,
        concat!("Alignment of ", stringify!(eqs_array_t))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).ptr) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_array_t),
            "::",
            stringify!(ptr)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).origin) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_array_t),
            "::",
            stringify!(origin)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).data) as usize - ptr as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_array_t),
            "::",
            stringify!(data)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).shape) as usize - ptr as usize },
        24usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_array_t),
            "::",
            stringify!(shape)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).reshape) as usize - ptr as usize },
        32usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_array_t),
            "::",
            stringify!(reshape)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).swap_axes) as usize - ptr as usize },
        40usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_array_t),
            "::",
            stringify!(swap_axes)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).create) as usize - ptr as usize },
        48usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_array_t),
            "::",
            stringify!(create)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).copy) as usize - ptr as usize },
        56usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_array_t),
            "::",
            stringify!(copy)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).destroy) as usize - ptr as usize },
        64usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_array_t),
            "::",
            stringify!(destroy)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).move_samples_from) as usize - ptr as usize },
        72usize,
        concat!(
            "Offset of field: ",
            stringify!(eqs_array_t),
            "::",
            stringify!(move_samples_from)
        )
    );
}
pub type eqs_create_array_callback_t = ::std::option::Option<
    unsafe extern "C" fn(
        shape: *const usize,
        shape_count: usize,
        array: *mut eqs_array_t,
    ) -> eqs_status_t,
>;
extern "C" {
    pub fn eqs_disable_panic_printing();
    pub fn eqs_version() -> *const ::std::os::raw::c_char;
    pub fn eqs_last_error() -> *const ::std::os::raw::c_char;
    #[must_use]
    pub fn eqs_labels_position(
        labels: eqs_labels_t,
        values: *const i32,
        values_count: usize,
        result: *mut i64,
    ) -> eqs_status_t;
    #[must_use]
    pub fn eqs_labels_create(labels: *mut eqs_labels_t) -> eqs_status_t;
    #[must_use]
    pub fn eqs_labels_clone(labels: eqs_labels_t, clone: *mut eqs_labels_t) -> eqs_status_t;
    #[must_use]
    pub fn eqs_labels_union(
        first: eqs_labels_t,
        second: eqs_labels_t,
        result: *mut eqs_labels_t,
        first_mapping: *mut i64,
        first_mapping_count: usize,
        second_mapping: *mut i64,
        second_mapping_count: usize,
    ) -> eqs_status_t;
    #[must_use]
    pub fn eqs_labels_intersection(
        first: eqs_labels_t,
        second: eqs_labels_t,
        result: *mut eqs_labels_t,
        first_mapping: *mut i64,
        first_mapping_count: usize,
        second_mapping: *mut i64,
        second_mapping_count: usize,
    ) -> eqs_status_t;
    #[must_use]
    pub fn eqs_labels_free(labels: *mut eqs_labels_t) -> eqs_status_t;
    #[must_use]
    pub fn eqs_register_data_origin(
        name: *const ::std::os::raw::c_char,
        origin: *mut eqs_data_origin_t,
    ) -> eqs_status_t;
    #[must_use]
    pub fn eqs_get_data_origin(
        origin: eqs_data_origin_t,
        buffer: *mut ::std::os::raw::c_char,
        buffer_size: usize,
    ) -> eqs_status_t;
    pub fn eqs_block(
        data: eqs_array_t,
        samples: eqs_labels_t,
        components: *const eqs_labels_t,
        components_count: usize,
        properties: eqs_labels_t,
    ) -> *mut eqs_block_t;
    #[must_use]
    pub fn eqs_block_free(block: *mut eqs_block_t) -> eqs_status_t;
    pub fn eqs_block_copy(block: *const eqs_block_t) -> *mut eqs_block_t;
    #[must_use]
    pub fn eqs_block_labels(
        block: *const eqs_block_t,
        axis: usize,
        labels: *mut eqs_labels_t,
    ) -> eqs_status_t;
    #[must_use]
    pub fn eqs_block_gradient(
        block: *mut eqs_block_t,
        parameter: *const ::std::os::raw::c_char,
        gradient: *mut *mut eqs_block_t,
    ) -> eqs_status_t;
    #[must_use]
    pub fn eqs_block_data(block: *mut eqs_block_t, data: *mut eqs_array_t) -> eqs_status_t;
    #[must_use]
    pub fn eqs_block_add_gradient(
        block: *mut eqs_block_t,
        parameter: *const ::std::os::raw::c_char,
        gradient: *mut eqs_block_t,
    ) -> eqs_status_t;
    #[must_use]
    pub fn eqs_block_gradients_list(
        block: *const eqs_block_t,
        parameters: *mut *const *const ::std::os::raw::c_char,
        parameters_count: *mut usize,
    ) -> eqs_status_t;
    pub fn eqs_tensormap(
        keys: eqs_labels_t,
        blocks: *mut *mut eqs_block_t,
        blocks_count: usize,
    ) -> *mut eqs_tensormap_t;
    #[must_use]
    pub fn eqs_tensormap_free(tensor: *mut eqs_tensormap_t) -> eqs_status_t;
    pub fn eqs_tensormap_copy(tensor: *const eqs_tensormap_t) -> *mut eqs_tensormap_t;
    #[must_use]
    pub fn eqs_tensormap_keys(
        tensor: *const eqs_tensormap_t,
        keys: *mut eqs_labels_t,
    ) -> eqs_status_t;
    #[must_use]
    pub fn eqs_tensormap_block_by_id(
        tensor: *mut eqs_tensormap_t,
        block: *mut *mut eqs_block_t,
        index: usize,
    ) -> eqs_status_t;
    #[must_use]
    pub fn eqs_tensormap_blocks_matching(
        tensor: *const eqs_tensormap_t,
        block_indexes: *mut usize,
        count: *mut usize,
        selection: eqs_labels_t,
    ) -> eqs_status_t;
    pub fn eqs_tensormap_keys_to_properties(
        tensor: *const eqs_tensormap_t,
        keys_to_move: eqs_labels_t,
        sort_samples: bool,
    ) -> *mut eqs_tensormap_t;
    pub fn eqs_tensormap_components_to_properties(
        tensor: *mut eqs_tensormap_t,
        dimensions: *const *const ::std::os::raw::c_char,
        dimensions_count: usize,
    ) -> *mut eqs_tensormap_t;
    pub fn eqs_tensormap_keys_to_samples(
        tensor: *const eqs_tensormap_t,
        keys_to_move: eqs_labels_t,
        sort_samples: bool,
    ) -> *mut eqs_tensormap_t;
    pub fn eqs_tensormap_load(
        path: *const ::std::os::raw::c_char,
        create_array: eqs_create_array_callback_t,
    ) -> *mut eqs_tensormap_t;
    #[must_use]
    pub fn eqs_tensormap_save(
        path: *const ::std::os::raw::c_char,
        tensor: *const eqs_tensormap_t,
    ) -> eqs_status_t;
}
