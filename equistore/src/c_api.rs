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
#[doc = " Status type returned by all functions in the C API.\n\n The value 0 (`EQS_SUCCESS`) is used to indicate successful operations,\n positive values are used by this library to indicate errors, while negative\n values are reserved for users of this library to indicate their own errors\n in callbacks."]
pub type eqs_status_t = i32;
#[doc = " A set of labels used to carry metadata associated with a tensor map.\n\n This is similar to a list of `count` named tuples, but stored as a 2D array\n of shape `(count, size)`, with a set of names associated with the columns of\n this array (often called *dimensions*). Each row/entry in this array is\n unique, and they are often (but not always) sorted in lexicographic order.\n\n `eqs_labels_t` with a non-NULL `internal_ptr_` correspond to a\n reference-counted Rust data structure, which allow for fast lookup inside\n the labels with `eqs_labels_positions`."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct eqs_labels_t {
    #[doc = " internal: pointer to the rust `Labels` struct if any, null otherwise"]
    pub internal_ptr_: *mut ::std::os::raw::c_void,
    #[doc = " Names of the dimensions composing this set of labels. There are `size`\n elements in this array, each being a NULL terminated UTF-8 string."]
    pub names: *const *const ::std::os::raw::c_char,
    #[doc = " Pointer to the first element of a 2D row-major array of 32-bit signed\n integer containing the values taken by the different dimensions in\n `names`. Each row has `size` elements, and there are `count` rows in\n total."]
    pub values: *const i32,
    #[doc = " Number of dimensions/size of a single entry in the set of labels"]
    pub size: usize,
    #[doc = " Number entries in the set of labels"]
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
#[doc = " A single 64-bit integer representing a data origin (numpy ndarray, rust\n ndarray, torch tensor, fortran array, ...)."]
pub type eqs_data_origin_t = u64;
#[doc = " Representation of a single sample moved from an array to another one"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct eqs_sample_mapping_t {
    #[doc = " index of the moved sample in the input array"]
    pub input: usize,
    #[doc = " index of the moved sample in the output array"]
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
#[doc = " `eqs_array_t` manages n-dimensional arrays used as data in a block or tensor\n map. The array itself if opaque to this library and can come from multiple\n sources: Rust program, a C/C++ program, a Fortran program, Python with numpy\n or torch. The data does not have to live on CPU, or even on the same machine\n where this code is executed.\n\n This struct contains a C-compatible manual implementation of a virtual table\n (vtable, i.e. trait in Rust, pure virtual class in C++); allowing\n manipulation of the array in an opaque way.\n\n **WARNING**: all function implementations **MUST** be thread-safe, and can\n be called from multiple threads at the same time. The `eqs_array_t` itself\n might be moved from one thread to another."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct eqs_array_t {
    #[doc = " User-provided data should be stored here, it will be passed as the\n first parameter to all function pointers below."]
    pub ptr: *mut ::std::os::raw::c_void,
    #[doc = " This function needs to store the \"data origin\" for this array in\n `origin`. Users of `eqs_array_t` should register a single data\n origin with `eqs_register_data_origin`, and use it for all compatible\n arrays."]
    pub origin: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            origin: *mut eqs_data_origin_t,
        ) -> eqs_status_t,
    >,
    #[doc = " Get a pointer to the underlying data storage.\n\n This function is allowed to fail if the data is not accessible in RAM,\n not stored as 64-bit floating point values, or not stored as a\n C-contiguous array."]
    pub data: ::std::option::Option<
        unsafe extern "C" fn(
            array: *mut ::std::os::raw::c_void,
            data: *mut *mut f64,
        ) -> eqs_status_t,
    >,
    #[doc = " Get the shape of the array managed by this `eqs_array_t` in the `*shape`\n pointer, and the number of dimension (size of the `*shape` array) in\n `*shape_count`."]
    pub shape: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            shape: *mut *const usize,
            shape_count: *mut usize,
        ) -> eqs_status_t,
    >,
    #[doc = " Change the shape of the array managed by this `eqs_array_t` to the given\n `shape`. `shape_count` must contain the number of elements in the\n `shape` array"]
    pub reshape: ::std::option::Option<
        unsafe extern "C" fn(
            array: *mut ::std::os::raw::c_void,
            shape: *const usize,
            shape_count: usize,
        ) -> eqs_status_t,
    >,
    #[doc = " Swap the axes `axis_1` and `axis_2` in this `array`."]
    pub swap_axes: ::std::option::Option<
        unsafe extern "C" fn(
            array: *mut ::std::os::raw::c_void,
            axis_1: usize,
            axis_2: usize,
        ) -> eqs_status_t,
    >,
    #[doc = " Create a new array with the same options as the current one (data type,\n data location, etc.) and the requested `shape`; and store it in\n `new_array`. The number of elements in the `shape` array should be given\n in `shape_count`.\n\n The new array should be filled with zeros."]
    pub create: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            shape: *const usize,
            shape_count: usize,
            new_array: *mut eqs_array_t,
        ) -> eqs_status_t,
    >,
    #[doc = " Make a copy of this `array` and return the new array in `new_array`.\n\n The new array is expected to have the same data origin and parameters\n (data type, data location, etc.)"]
    pub copy: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            new_array: *mut eqs_array_t,
        ) -> eqs_status_t,
    >,
    #[doc = " Remove this array and free the associated memory. This function can be\n set to `NULL` is there is no memory management to do."]
    pub destroy: ::std::option::Option<unsafe extern "C" fn(array: *mut ::std::os::raw::c_void)>,
    #[doc = " Set entries in the `output` array (the current array) taking data from\n the `input` array. The `output` array is guaranteed to be created by\n calling `eqs_array_t::create` with one of the arrays in the same block\n or tensor map as the `input`.\n\n The `samples` array of size `samples_count` indicate where the data\n should be moved from `input` to `output`.\n\n This function should copy data from `input[samples[i].input, ..., :]` to\n `array[samples[i].output, ..., property_start:property_end]` for `i` up\n to `samples_count`. All indexes are 0-based."]
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
#[doc = " Function pointer to create a new `eqs_array_t` when de-serializing tensor\n maps.\n\n This function gets the `shape` of the array (the `shape` contains\n `shape_count` elements) and should fill `array` with a new valid\n `eqs_array_t` or return non-zero `eqs_status_t`.\n\n The newly created array should contains 64-bit floating points (`double`)\n data, and live on CPU, since equistore will use `eqs_array_t.data` to get\n the data pointer and write to it."]
pub type eqs_create_array_callback_t = ::std::option::Option<
    unsafe extern "C" fn(
        shape: *const usize,
        shape_count: usize,
        array: *mut eqs_array_t,
    ) -> eqs_status_t,
>;
extern "C" {
    #[doc = " Disable printing of the message to stderr when some Rust code reach a panic.\n\n All panics from Rust code are caught anyway and translated to an error\n status code, and the message is stored and accessible through\n `eqs_last_error`. To print the error message and Rust backtrace anyway,\n users can set the `RUST_BACKTRACE` environment variable to 1."]
    pub fn eqs_disable_panic_printing();
    #[doc = " Get the version of the core equistore library as a string.\n\n This version should follow the `<major>.<minor>.<patch>[-<dev>]` format."]
    pub fn eqs_version() -> *const ::std::os::raw::c_char;
    #[doc = " Get the last error message that was created on the current thread.\n\n @returns the last error message, as a NULL-terminated string"]
    pub fn eqs_last_error() -> *const ::std::os::raw::c_char;
    #[must_use]
    #[doc = " Get the position of the entry defined by the `values` array in the given set\n of `labels`. This operation is only available if the labels correspond to a\n set of Rust Labels (i.e. `labels.internal_ptr_` is not NULL).\n\n @param labels set of labels with an associated Rust data structure\n @param values array containing the label to lookup\n @param values_count size of the values array\n @param result position of the values in the labels or -1 if the values\n               were not found\n\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_labels_position(
        labels: eqs_labels_t,
        values: *const i32,
        values_count: usize,
        result: *mut i64,
    ) -> eqs_status_t;
    #[must_use]
    #[doc = " Finish the creation of `eqs_labels_t` by associating it to Rust-owned\n labels.\n\n This allows using the `eqs_labels_positions` and `eqs_labels_clone`\n functions on the `eqs_labels_t`.\n\n This function allocates memory which must be released `eqs_labels_free` when\n you don't need it anymore.\n\n @param labels new set of labels containing pointers to user-managed memory\n        on input, and pointers to Rust-managed memory on output.\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_labels_create(labels: *mut eqs_labels_t) -> eqs_status_t;
    #[must_use]
    #[doc = " Make a copy of `labels` inside `clone`.\n\n Since `eqs_labels_t` are immutable, the copy is actually just a reference\n count increase, and as such should not be an expensive operation.\n\n `eqs_labels_free` must be used with `clone` to decrease the reference count\n and release the memory when you don't need it anymore.\n\n @param labels set of labels with an associated Rust data structure\n @param clone empty labels, on output will contain a copy of `labels`\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_labels_clone(labels: eqs_labels_t, clone: *mut eqs_labels_t) -> eqs_status_t;
    #[must_use]
    #[doc = " Decrease the reference count of `labels`, and release the corresponding\n memory once the reference count reaches 0.\n\n @param labels set of labels with an associated Rust data structure\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_labels_free(labels: *mut eqs_labels_t) -> eqs_status_t;
    #[must_use]
    #[doc = " Register a new data origin with the given `name`. Calling this function\n multiple times with the same name will give the same `eqs_data_origin_t`.\n\n @param name name of the data origin as an UTF-8 encoded NULL-terminated string\n @param origin pointer to an `eqs_data_origin_t` where the origin will be stored\n\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_register_data_origin(
        name: *const ::std::os::raw::c_char,
        origin: *mut eqs_data_origin_t,
    ) -> eqs_status_t;
    #[must_use]
    #[doc = " Get the name used to register a given data `origin` in the given `buffer`\n\n @param origin pre-registered data origin\n @param buffer buffer to be filled with the data origin name. The origin name\n               will be written  as an UTF-8 encoded, NULL-terminated string\n @param buffer_size size of the buffer\n\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_get_data_origin(
        origin: eqs_data_origin_t,
        buffer: *mut ::std::os::raw::c_char,
        buffer_size: usize,
    ) -> eqs_status_t;
    #[doc = " Create a new `eqs_block_t` with the given `data` and `samples`, `components`\n and `properties` labels.\n\n The memory allocated by this function and the blocks should be released\n using `eqs_block_free`, or moved into a tensor map using `eqs_tensormap`.\n\n @param data array handle containing the data for this block. The block takes\n             ownership of the array, and will release it with\n             `array.destroy(array.ptr)` when it no longer needs it.\n @param samples sample labels corresponding to the first dimension of the data\n @param components array of component labels corresponding to intermediary\n                   dimensions of the data\n @param components_count number of entries in the `components` array\n @param properties property labels corresponding to the last dimension of the data\n\n @returns A pointer to the newly allocated block, or a `NULL` pointer in\n          case of error. In case of error, you can use `eqs_last_error()`\n          to get the error message."]
    pub fn eqs_block(
        data: eqs_array_t,
        samples: eqs_labels_t,
        components: *const eqs_labels_t,
        components_count: usize,
        properties: eqs_labels_t,
    ) -> *mut eqs_block_t;
    #[must_use]
    #[doc = " Free the memory associated with a `block` previously created with\n `eqs_block`.\n\n If `block` is `NULL`, this function does nothing.\n\n @param block pointer to an existing block, or `NULL`\n\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_block_free(block: *mut eqs_block_t) -> eqs_status_t;
    #[doc = " Make a copy of an `eqs_block_t`.\n\n The memory allocated by this function and the blocks should be released\n using `eqs_block_free`, or moved into a tensor map using `eqs_tensormap`.\n\n @param block existing block to copy\n\n @returns A pointer to the newly allocated block, or a `NULL` pointer in\n          case of error. In case of error, you can use `eqs_last_error()`\n          to get the error message."]
    pub fn eqs_block_copy(block: *const eqs_block_t) -> *mut eqs_block_t;
    #[must_use]
    #[doc = " Get the set of labels of the requested `kind` from this `block`.\n\n The `values_gradients` parameter controls whether this function looks up\n labels for `\"values\"` or one of the gradients in this block.\n\n This function allocates memory for `labels` which must be released\n `eqs_labels_free` when you don't need it anymore.\n\n @param block pointer to an existing block\n @param values_gradients either `\"values\"` or the name of gradients to lookup\n @param axis axis/dimension of the data array for which you need the labels\n @param labels pointer to an empty `eqs_labels_t` that will be set to the\n               requested labels\n\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_block_labels(
        block: *const eqs_block_t,
        values_gradients: *const ::std::os::raw::c_char,
        axis: usize,
        labels: *mut eqs_labels_t,
    ) -> eqs_status_t;
    #[must_use]
    #[doc = " Get the array handle for either values or one of the gradient in this `block`.\n\n The `values_gradients` parameter controls whether this function looks up\n labels for `\"values\"` or one of the gradients in this block.\n\n @param block pointer to an existing block\n @param values_gradients either `\"values\"` or the name of gradients to lookup\n @param data pointer to an empty `eqs_array_t` that will be set to the\n             requested array\n\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_block_data(
        block: *mut eqs_block_t,
        values_gradients: *const ::std::os::raw::c_char,
        data: *mut eqs_array_t,
    ) -> eqs_status_t;
    #[must_use]
    #[doc = " Add a new gradient to this `block` with the given `name`.\n\n @param block pointer to an existing block\n @param data array containing the gradient data. The block takes\n                 ownership of the array, and will release it with\n                 `array.destroy(array.ptr)` when it no longer needs it.\n @param parameter name of the gradient as a NULL-terminated UTF-8 string.\n                  This is usually the parameter used when taking derivatives\n                  (e.g. `\"positions\"`, `\"cell\"`, etc.)\n @param samples sample labels for the gradient array. The components and\n                property labels are supposed to match the values in this block\n @param components array of component labels corresponding to intermediary\n                   dimensions of the data\n @param components_count number of entries in the `components` array\n\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_block_add_gradient(
        block: *mut eqs_block_t,
        parameter: *const ::std::os::raw::c_char,
        data: eqs_array_t,
        samples: eqs_labels_t,
        components: *const eqs_labels_t,
        components_count: usize,
    ) -> eqs_status_t;
    #[must_use]
    #[doc = " Get a list of all gradients defined in this `block` in the `parameters` array.\n\n @param block pointer to an existing block\n @param parameters will be set to the first element of an array of\n                   NULL-terminated UTF-8 strings containing all the\n                   parameters for which a gradient exists in the block\n @param parameters_count will be set to the number of elements in `parameters`\n\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_block_gradients_list(
        block: *const eqs_block_t,
        parameters: *mut *const *const ::std::os::raw::c_char,
        parameters_count: *mut usize,
    ) -> eqs_status_t;
    #[doc = " Create a new `eqs_tensormap_t` with the given `keys` and `blocks`.\n `blocks_count` must be set to the number of entries in the blocks array.\n\n The new tensor map takes ownership of the blocks, which should not be\n released separately.\n\n The memory allocated by this function and the blocks should be released\n using `eqs_tensormap_free`.\n\n @param keys labels containing the keys associated with each block\n @param blocks pointer to the first element of an array of blocks\n @param blocks_count number of elements in the `blocks` array\n\n @returns A pointer to the newly allocated tensor map, or a `NULL` pointer in\n          case of error. In case of error, you can use `eqs_last_error()`\n          to get the error message."]
    pub fn eqs_tensormap(
        keys: eqs_labels_t,
        blocks: *mut *mut eqs_block_t,
        blocks_count: usize,
    ) -> *mut eqs_tensormap_t;
    #[must_use]
    #[doc = " Free the memory associated with a `tensor` previously created with\n `eqs_tensormap`.\n\n If `tensor` is `NULL`, this function does nothing.\n\n @param tensor pointer to an existing tensor map, or `NULL`\n\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_tensormap_free(tensor: *mut eqs_tensormap_t) -> eqs_status_t;
    #[doc = " Make a copy of an `eqs_tensormap_t`.\n\n The memory allocated by this function and the blocks should be released\n using `eqs_tensormap_free`.\n\n @param tensor existing tensor to copy\n\n @returns A pointer to the newly allocated tensor, or a `NULL` pointer in\n          case of error. In case of error, you can use `eqs_last_error()`\n          to get the error message."]
    pub fn eqs_tensormap_copy(tensor: *const eqs_tensormap_t) -> *mut eqs_tensormap_t;
    #[must_use]
    #[doc = " Get the keys for the given `tensor` map.\n\n This function allocates memory for `keys` which must be released\n `eqs_labels_free` when you don't need it anymore.\n\n @param tensor pointer to an existing tensor map\n @param keys pointer to be filled with the keys of the tensor map\n\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_tensormap_keys(
        tensor: *const eqs_tensormap_t,
        keys: *mut eqs_labels_t,
    ) -> eqs_status_t;
    #[must_use]
    #[doc = " Get a pointer to the `index`-th block in this tensor map.\n\n The block memory is still managed by the tensor map, this block should not\n be freed. The block is invalidated when the tensor map is freed with\n `eqs_tensormap_free` or the set of keys is modified by calling one\n of the `eqs_tensormap_keys_to_XXX` function.\n\n @param tensor pointer to an existing tensor map\n @param block pointer to be filled with a block\n @param index index of the block to get\n\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_tensormap_block_by_id(
        tensor: *mut eqs_tensormap_t,
        block: *mut *mut eqs_block_t,
        index: usize,
    ) -> eqs_status_t;
    #[must_use]
    #[doc = " Get indices of the blocks in this `tensor` corresponding to the given\n `selection`. The `selection` should have a subset of the names/dimensions of\n the keys for this tensor map, and only one entry, describing the requested\n blocks.\n\n When calling this function, `*count` should contain the number of entries in\n `block_indexes`. When the function returns successfully, `*count` will\n contain the number of blocks matching the selection, i.e. how many values\n were written to `block_indexes`.\n\n @param tensor pointer to an existing tensor map\n @param block_indexes array to be filled with indexes of blocks in the tensor\n                      map matching the `selection`\n @param count number of entries in `block_indexes`\n @param selection labels with a single entry describing which blocks are requested\n\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_tensormap_blocks_matching(
        tensor: *const eqs_tensormap_t,
        block_indexes: *mut usize,
        count: *mut usize,
        selection: eqs_labels_t,
    ) -> eqs_status_t;
    #[doc = " Merge blocks with the same value for selected keys dimensions along the\n property axis.\n\n The dimensions (names) of `keys_to_move` will be moved from the keys to\n the property labels, and blocks with the same remaining keys dimensions\n will be merged together along the property axis.\n\n If `keys_to_move` does not contains any entries (`keys_to_move.count\n == 0`), then the new property labels will contain entries corresponding\n to the merged blocks only. For example, merging a block with key `a=0`\n and properties `p=1, 2` with a block with key `a=2` and properties `p=1,\n 3` will produce a block with properties `a, p = (0, 1), (0, 2), (2, 1),\n (2, 3)`.\n\n If `keys_to_move` contains entries, then the property labels must be the\n same for all the merged blocks. In that case, the merged property labels\n will contains each of the entries of `keys_to_move` and then the current\n property labels. For example, using `a=2, 3` in `keys_to_move`, and\n blocks with properties `p=1, 2` will result in `a, p = (2, 1), (2, 2),\n (3, 1), (3, 2)`.\n\n The new sample labels will contains all of the merged blocks sample\n labels. The order of the samples is controlled by `sort_samples`. If\n `sort_samples` is true, samples are re-ordered to keep them\n lexicographically sorted. Otherwise they are kept in the order in which\n they appear in the blocks.\n\n The result is a new tensor map, which should be freed with `eqs_tensormap_free`.\n\n @param tensor pointer to an existing tensor map\n @param keys_to_move description of the keys to move\n @param sort_samples whether to sort the samples lexicographically after\n                     merging blocks\n\n @returns A pointer to the newly allocated tensor map, or a `NULL` pointer in\n          case of error. In case of error, you can use `eqs_last_error()`\n          to get the error message."]
    pub fn eqs_tensormap_keys_to_properties(
        tensor: *const eqs_tensormap_t,
        keys_to_move: eqs_labels_t,
        sort_samples: bool,
    ) -> *mut eqs_tensormap_t;
    #[doc = " Move the given dimensions from the component labels to the property labels\n for each block in this tensor map.\n\n `dimensions` must be an array of `dimensions_count` NULL-terminated strings,\n encoded as UTF-8.\n\n @param tensor pointer to an existing tensor map\n @param dimensions names of the key dimensions to move to the properties\n @param dimensions_count number of entries in the `dimensions` array\n\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_tensormap_components_to_properties(
        tensor: *mut eqs_tensormap_t,
        dimensions: *const *const ::std::os::raw::c_char,
        dimensions_count: usize,
    ) -> *mut eqs_tensormap_t;
    #[doc = " Merge blocks with the same value for selected keys dimensions along the\n samples axis.\n\n The dimensions (names) of `keys_to_move` will be moved from the keys to\n the sample labels, and blocks with the same remaining keys dimensions\n will be merged together along the sample axis.\n\n `keys_to_move` must be empty (`keys_to_move.count == 0`), and the new\n sample labels will contain entries corresponding to the merged blocks'\n keys.\n\n The new sample labels will contains all of the merged blocks sample\n labels. The order of the samples is controlled by `sort_samples`. If\n `sort_samples` is true, samples are re-ordered to keep them\n lexicographically sorted. Otherwise they are kept in the order in which\n they appear in the blocks.\n\n This function is only implemented if all merged block have the same\n property labels.\n\n @param tensor pointer to an existing tensor map\n @param keys_to_move description of the keys to move\n @param sort_samples whether to sort the samples lexicographically after\n                     merging blocks or not\n\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_tensormap_keys_to_samples(
        tensor: *const eqs_tensormap_t,
        keys_to_move: eqs_labels_t,
        sort_samples: bool,
    ) -> *mut eqs_tensormap_t;
    #[doc = " Load a tensor map from the file at the given path.\n\n Arrays for the values and gradient data will be created with the given\n `create_array` callback, and filled by this function with the corresponding\n data.\n\n The memory allocated by this function should be released using\n `eqs_tensormap_free`.\n\n `TensorMap` are serialized using numpy's `.npz` format, i.e. a ZIP file\n without compression (storage method is STORED), where each file is stored as\n a `.npy` array. Both the ZIP and NPY format are well documented:\n\n - ZIP: <https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT>\n - NPY: <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>\n\n We add other restriction on top of these formats when saving/loading data.\n First, `Labels` instances are saved as structured array, see the `labels`\n module for more information. Only 32-bit integers are supported for Labels,\n and only 64-bit floats are supported for data (values and gradients).\n\n Second, the path of the files in the archive also carry meaning. The keys of\n the `TensorMap` are stored in `/keys.npy`, and then different blocks are\n stored as\n\n ```bash\n /  blocks / <block_id>  / values / samples.npy\n                         / values / components  / 0.npy\n                                                / <...>.npy\n                                                / <n_components>.npy\n                         / values / properties.npy\n                         / values / data.npy\n\n                         # optional sections for gradients, one by parameter\n                         /   gradients / <parameter> / samples.npy\n                                                     /   components  / 0.npy\n                                                                     / <...>.npy\n                                                                     / <n_components>.npy\n                                                     /   data.npy\n ```\n\n @param path path to the file as a NULL-terminated UTF-8 string\n @param create_array callback function that will be used to create data\n                     arrays inside each block\n\n @returns A pointer to the newly allocated tensor map, or a `NULL` pointer in\n          case of error. In case of error, you can use `eqs_last_error()`\n          to get the error message."]
    pub fn eqs_tensormap_load(
        path: *const ::std::os::raw::c_char,
        create_array: eqs_create_array_callback_t,
    ) -> *mut eqs_tensormap_t;
    #[must_use]
    #[doc = " Save a tensor map to the file at the given path.\n\n If the file already exists, it is overwritten.\n\n @param path path to the file as a NULL-terminated UTF-8 string\n @param tensor tensor map to save to the file\n\n @returns The status code of this operation. If the status is not\n          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full\n          error message."]
    pub fn eqs_tensormap_save(
        path: *const ::std::os::raw::c_char,
        tensor: *const eqs_tensormap_t,
    ) -> eqs_status_t;
}
