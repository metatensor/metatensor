use std::sync::Arc;
use std::os::raw::c_char;
use std::ffi::CStr;

use crate::{TensorBlock, Error, eqs_array_t};

use super::labels::{eqs_labels_t, rust_to_eqs_labels, eqs_labels_to_rust};

use super::{catch_unwind, eqs_status_t};

/// Basic building block for tensor map. A single block contains a n-dimensional
/// `eqs_array_t`, and n sets of `eqs_labels_t` (one for each dimension).
///
/// A block can also contain gradients of the values with respect to a variety
/// of parameters. In this case, each gradient has a separate set of sample
/// and component labels but share the property labels with the values.
#[allow(non_camel_case_types)]
pub struct eqs_block_t(TensorBlock);

impl std::ops::Deref for eqs_block_t {
    type Target = TensorBlock;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for eqs_block_t {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl eqs_block_t {
    pub(super) fn into_block(self) -> TensorBlock {
        self.0
    }
}


/// Create a new `eqs_block_t` with the given `data` and `samples`, `components`
/// and `properties` labels.
///
/// The memory allocated by this function and the blocks should be released
/// using `eqs_block_free`, or moved into a tensor map using `eqs_tensormap`.
///
/// @param data array handle containing the data for this block. The block takes
///             ownership of the array, and will release it with
///             `array.destroy(array.ptr)` when it no longer needs it.
/// @param samples sample labels corresponding to the first dimension of the data
/// @param components array of component labels corresponding to intermediary
///                   dimensions of the data
/// @param components_count number of entries in the `components` array
/// @param properties property labels corresponding to the last dimension of the data
///
/// @returns A pointer to the newly allocated block, or a `NULL` pointer in
///          case of error. In case of error, you can use `eqs_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern fn eqs_block(
    data: eqs_array_t,
    samples: eqs_labels_t,
    components: *const eqs_labels_t,
    components_count: usize,
    properties: eqs_labels_t,
) -> *mut eqs_block_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        let samples = eqs_labels_to_rust(&samples)?;

        let mut rust_components = Vec::new();
        if components_count != 0 {
            check_pointers!(components);
            for component in std::slice::from_raw_parts(components, components_count) {
                rust_components.push(eqs_labels_to_rust(component)?);
            }
        }

        let properties = eqs_labels_to_rust(&properties)?;

        let block = TensorBlock::new(data, samples, rust_components, properties)?;
        let boxed = Box::new(eqs_block_t(block));

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
/// `eqs_block`.
///
/// If `block` is `NULL`, this function does nothing.
///
/// @param block pointer to an existing block, or `NULL`
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_block_free(
    block: *mut eqs_block_t,
) -> eqs_status_t {
    catch_unwind(|| {
        if !block.is_null() {
            std::mem::drop(Box::from_raw(block));
        }

        Ok(())
    })
}

/// Make a copy of an `eqs_block_t`.
///
/// The memory allocated by this function and the blocks should be released
/// using `eqs_block_free`, or moved into a tensor map using `eqs_tensormap`.
///
/// @param block existing block to copy
///
/// @returns A pointer to the newly allocated block, or a `NULL` pointer in
///          case of error. In case of error, you can use `eqs_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern fn eqs_block_copy(
    block: *const eqs_block_t,
) -> *mut eqs_block_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        check_pointers!(block);
        let new_block = (*block).try_clone()?;
        let boxed = Box::new(eqs_block_t(new_block));

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


/// Get the set of labels from this `block`.
///
/// This function allocates memory for `labels` which must be released
/// `eqs_labels_free` when you don't need it anymore.
///
/// @param block pointer to an existing block
/// @param axis axis/dimension of the data array for which you need the labels
/// @param labels pointer to an empty `eqs_labels_t` that will be set to the
///        `block`'s labels
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_block_labels(
    block: *const eqs_block_t,
    axis: usize,
    labels: *mut eqs_labels_t,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(block, labels);

        if (*labels).is_rust() {
            return Err(Error::InvalidParameter(
                "these labels are already allocated, call eqs_labels_free first".into()
            ));
        }

        let block = &(*block);
        let n_components = block.components.len();

        let rust_labels = if axis == 0 {
            &block.samples
        } else if axis - 1 < n_components {
            // component labels
            &block.components[axis - 1]
        } else if axis == n_components + 1 {
            // property labels
            &block.properties
        } else {
            return Err(Error::InvalidParameter(format!(
                "tried to get the labels for axis {}, but we only have {} axes for this block",
                axis, n_components + 2
            )));
        };

        *labels = rust_to_eqs_labels(Arc::clone(rust_labels));

        Ok(())
    })
}

fn _get_gradient<'a>(
    block: &'a mut TensorBlock,
    parameter: &str
) -> Result<&'a mut TensorBlock, Error> {
    let gradient_block =
        if let Some((first, rest)) = parameter.split_once('/') {
            _get_gradient(
                block.gradient_mut(first).ok_or_else(|| Error::InvalidParameter(format!(
                    "can not find gradients with respect to '{}' in this block", first
                )))?,
                rest
            )?
        } else {
            block.gradient_mut(parameter).ok_or_else(|| Error::InvalidParameter(format!(
                "can not find gradients with respect to '{}' in this block", parameter
            )))?
        };

    Ok(gradient_block)
}


/// Get one of the gradients in this `block`.
///
/// The gradient memory is still managed by the block, the returned
/// `eqs_block_t*` should not be freed. The gradient pointer is invalidated if
/// more gradients are added to the parent block, or if the parent block is
/// freed with `eqs_block_free`.
///
/// @param block pointer to an existing block
/// @param parameter the name of the gradient to be extracted
/// @param gradient pointer to an empty `eqs_block_t` pointer that will be
///        overwritten to the requested gradient
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full error
///          message.
#[no_mangle]
pub unsafe extern fn eqs_block_gradient(
    block: *mut eqs_block_t,
    parameter: *const c_char,
    gradient: *mut *mut eqs_block_t
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(block, parameter);
        let parameter = CStr::from_ptr(parameter).to_str().unwrap();

        let gradient_rust = (*block).gradient_mut(parameter).ok_or_else(|| {
            Error::InvalidParameter(format!(
                "can not find gradients with respect to '{}' in this block", parameter
            ))
        })?;
        (*gradient) = (gradient_rust as *mut TensorBlock).cast();

        Ok(())
    })
}


/// Get the array handle for the values in this `block`.
///
/// @param block pointer to an existing block
/// @param data pointer to an empty `eqs_array_t` that will be set to the
///             requested array
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_block_data(
    block: *mut eqs_block_t,
    data: *mut eqs_array_t,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(block, data);
        *data = (*block).values.raw_copy();
        Ok(())
    })
}


/// Add a new gradient to this `block` with the given `name`.
///
/// The block takes ownership of the gradient, which should not be released
/// separately.
///
/// @param block pointer to an existing block
/// @param parameter name of the gradient as a NULL-terminated UTF-8 string.
///                  This is usually the parameter used when taking derivatives
///                  (e.g. `"positions"`, `"cell"`, etc.)
/// @param gradient a block whose values contain the gradients with respect to
///                 the `parameter`. The labels of the `gradient` should be
///                 organized as follows: its `samples` must contain `"sample"`
///                 as the first label, which establishes a correspondence with
///                 the `samples` of the original `block`; its components must
///                 contain at least the same components as the original
///                 `TensorBlock`, with any additional component coming before
///                 those; its properties must match those of the original
///                 `block`.
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_block_add_gradient(
    block: *mut eqs_block_t,
    parameter: *const c_char,
    gradient: *mut eqs_block_t,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(block, parameter);
        // TODO: add a check that the block is not already part of a tensor map?
        let parameter = CStr::from_ptr(parameter).to_str().unwrap();

        // move the gradient out of the pointer
        let gradient = Box::from_raw(gradient).into_block();

        (*block).add_gradient(parameter, gradient)?;
        Ok(())
    })
}

/// Get a list of all gradients defined in this `block` in the `parameters` array.
///
/// @param block pointer to an existing block
/// @param parameters will be set to the first element of an array of
///                   NULL-terminated UTF-8 strings containing all the
///                   parameters for which a gradient exists in the block
/// @param parameters_count will be set to the number of elements in `parameters`
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_block_gradients_list(
    block: *const eqs_block_t,
    parameters: *mut *const *const c_char,
    parameters_count: *mut usize
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(block, parameters, parameters_count);

        let list = (*block).gradient_parameters_c();
        (*parameters_count) = list.len();

        (*parameters) = if list.is_empty() {
            std::ptr::null()
        } else {
            list.as_ptr().cast()
        };
        Ok(())
    })
}
