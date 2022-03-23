use std::sync::Arc;
use std::os::raw::c_char;
use std::ffi::CStr;
use std::convert::{TryFrom, TryInto};

use crate::{Block, Labels, Error, aml_array_t};

use super::labels::{aml_labels_t, aml_label_kind};

use super::{catch_unwind, aml_status_t};

/// Basic building block for descriptor. A single block contains a 3-dimensional
/// `aml_array_t`, and three sets of `aml_labels_t` (one for each dimension).
///
/// A block can also contain gradients of the values with respect to a variety
/// of parameters. In this case, each gradient has a separate set of samples,
/// but share the same components and feature labels as the values.
#[allow(non_camel_case_types)]
pub struct aml_block_t(Block);

impl std::ops::Deref for aml_block_t {
    type Target = Block;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for aml_block_t {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl aml_block_t {
    pub(super) fn block(self) -> Block {
        self.0
    }
}


/// Create a new `aml_block_t` with the given `data` and `samples`, `components`
/// and `features` labels.
///
/// The memory allocated by this function and the blocks should be released
/// using `aml_block_free`, or moved into a descriptor using `aml_descriptor`.
///
/// @param data array handle containing the data for this block. The block takes
///             ownership of the array, and will release it with
///             `array.destroy(array.ptr)` when it no longer needs it.
/// @param samples sample labels corresponding to the first dimension of the data
/// @param components component labels corresponding to the second dimension of the data
/// @param features feature labels corresponding to the third dimension of the data
///
/// @returns A pointer to the newly allocated block, or a `NULL` pointer in
///          case of error. In case of error, you can use `aml_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern fn aml_block(
    data: aml_array_t,
    samples: aml_labels_t,
    components: aml_labels_t,
    features: aml_labels_t,
) -> *mut aml_block_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        let samples = Labels::try_from(samples)?;
        let components = Labels::try_from(components)?;
        let features = Labels::try_from(features)?;

        let block = Block::new(data, samples, Arc::new(components), Arc::new(features))?;
        let boxed = Box::new(aml_block_t(block));

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *(unwind_wrapper.0) = Box::into_raw(boxed);
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}


/// Free the memory associated with a `block` previously created with
/// `aml_block`.
///
/// If `block` is `NULL`, this function does nothing.
///
/// @param block pointer to an existing block, or `NULL`
///
/// @returns The status code of this operation. If the status is not
///          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn aml_block_free(
    block: *mut aml_block_t,
) -> aml_status_t {
    catch_unwind(|| {
        if !block.is_null() {
            std::mem::drop(Box::from_raw(block));
        }

        Ok(())
    })
}


#[allow(clippy::doc_markdown)]
/// Get the set of labels of the requested `kind` from this `block`.
///
/// The `values_gradients` parameter controls whether this function looks up
/// labels for `"values"` or one of the gradients in this block.
///
/// The resulting `labels.values` points inside memory owned by the block, and
/// as such is only valid until the block is destroyed with `aml_block_free`, or
/// the containing descriptor is modified with one of the
/// `aml_descriptor_sparse_to_xxx` function.
///
/// @param block pointer to an existing block
/// @param values_gradients either `"values"` or the name of gradients to lookup
/// @param kind the kind of labels requested
/// @param labels pointer to an empty `aml_labels_t` that will be set to the
///               requested labels
///
/// @returns The status code of this operation. If the status is not
///          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn aml_block_labels(
    block: *const aml_block_t,
    values_gradients: *const c_char,
    kind: aml_label_kind,
    labels: *mut aml_labels_t,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(block, values_gradients, labels);

        let values_gradients = CStr::from_ptr(values_gradients).to_str().unwrap();
        let basic_block = match values_gradients {
            "values" => &(*block).values,
            gradients => {
                (*block).get_gradient(gradients).ok_or_else(|| Error::InvalidParameter(format!(
                    "can not find gradients with respect to '{}' in this block", gradients
                )))?
            }
        };

        let rust_labels = match kind {
            aml_label_kind::AML_SAMPLE_LABELS => basic_block.samples(),
            aml_label_kind::AML_COMPONENTS_LABELS => basic_block.components(),
            aml_label_kind::AML_FEATURE_LABELS => basic_block.features(),
        };

        *labels = rust_labels.try_into()?;

        Ok(())
    })
}


#[allow(clippy::doc_markdown)]
/// Get the array handle for either values or one of the gradient in this `block`.
///
/// The `values_gradients` parameter controls whether this function looks up
/// labels for `"values"` or one of the gradients in this block.
///
/// @param block pointer to an existing block
/// @param values_gradients either `"values"` or the name of gradients to lookup
/// @param data pointer to an empty `aml_array_t` that will be set to the
///             requested array
///
/// @returns The status code of this operation. If the status is not
///          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn aml_block_data(
    block: *const aml_block_t,
    values_gradients: *const c_char,
    data: *mut *const aml_array_t,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(block, values_gradients, data);

        let values_gradients = CStr::from_ptr(values_gradients).to_str().unwrap();
        let basic_block = match values_gradients {
            "values" => &(*block).values,
            gradients => {
                (*block).get_gradient(gradients).ok_or_else(|| Error::InvalidParameter(format!(
                    "can not find gradients with respect to '{}' in this block", gradients
                )))?
            }
        };

        *data = &basic_block.data as *const _;

        Ok(())
    })
}


/// Add a new gradient to this `block` with the given `name`.
///
/// @param block pointer to an existing block
/// @param name name of the gradient as a NULL-terminated UTF-8 string. This is
///             usually the parameter used when taking derivatives (e.g.
///             `"positions"`, `"cell"`, etc.)
/// @param samples sample labels for the gradient array. The components and
///                feature labels are supposed to match the values in this block
/// @param gradient array containing the gradient data. The block takes
///                 ownership of the array, and will release it with
///                 `array.destroy(array.ptr)` when it no longer needs it.
///
/// @returns The status code of this operation. If the status is not
///          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn aml_block_add_gradient(
    block: *mut aml_block_t,
    name: *const c_char,
    samples: aml_labels_t,
    gradient: aml_array_t,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(block, name);
        let name = CStr::from_ptr(name).to_str().unwrap();
        let samples = Labels::try_from(samples)?;
        (*block).add_gradient(name, samples, gradient)?;
        Ok(())
    })
}


#[allow(clippy::doc_markdown)]
/// Check if this `block` contains gradient with the given `name`.
///
/// @param block pointer to an existing block
/// @param name name of the gradient as a NULL-terminated UTF-8 string
/// @param has_gradient pointer to bool that will be set to `true` if this block
///                     contains the requested gradients, and `false` otherwise
///
/// @returns The status code of this operation. If the status is not
///          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn aml_block_has_gradient(
    block: *mut aml_block_t,
    name: *const c_char,
    has_gradient: *mut bool,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(block, name, has_gradient);
        let name = CStr::from_ptr(name).to_str().unwrap();
        *has_gradient = (*block).has_gradient(name);
        Ok(())
    })
}
