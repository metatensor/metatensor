use std::os::raw::c_char;
use std::ffi::CStr;
use std::convert::{TryFrom, TryInto};
use std::collections::BTreeSet;

use crate::{Descriptor, Labels, Block, Error};

use super::labels::aml_labels_t;
use super::blocks::aml_block_t;
use super::status::{aml_status_t, catch_unwind};

/// Opaque type representing a `Descriptor`.
#[allow(non_camel_case_types)]
pub struct aml_descriptor_t(Descriptor);

impl std::ops::Deref for aml_descriptor_t {
    type Target = Descriptor;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for aml_descriptor_t {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}


/// Create a new `aml_descriptor_t` with the given `sparse` labels and `blocks`.
/// `blocks_count` must be set to the number of entries in the blocks array.
///
/// The new descriptor takes ownership of the blocks, which should not be
/// released separately.
///
/// The memory allocated by this function and the blocks should be released
/// using `aml_descriptor_free`.
///
/// @param sparse sparse labels associated with each block
/// @param blocks pointer to the first element of an array of blocks
/// @param blocks_count number of elements in the `blocks` array
///
/// @returns A pointer to the newly allocated descriptor, or a `NULL` pointer in
///          case of error. In case of error, you can use `aml_last_error()`
///          to get the error message.
#[no_mangle]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern fn aml_descriptor(
    sparse: aml_labels_t,
    blocks: *mut *mut aml_block_t,
    blocks_count: u64,
) -> *mut aml_descriptor_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        let sparse = Labels::try_from(&sparse)?;

        let blocks_slice = std::slice::from_raw_parts_mut(blocks, blocks_count as usize);
        // check for uniqueness of the pointers: we don't want to move out
        // the same value twice
        if blocks_slice.iter().collect::<BTreeSet<_>>().len() != blocks_slice.len() {
            return Err(Error::InvalidParameter(
                "got the same block more than once when constructing a descriptor".into()
            ));
        }

        let blocks_vec = blocks_slice.iter_mut().map(|ptr| {
            // move out of the blocks pointers
            let block = Box::from_raw(*ptr).block();
            *ptr = std::ptr::null_mut();
            return block;
        }).collect();

        let descriptor = Descriptor::new(sparse, blocks_vec)?;
        let boxed = Box::new(aml_descriptor_t(descriptor));

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


/// Free the memory associated with a `descriptor` previously created with
/// `aml_descriptor`.
///
/// If `descriptor` is `NULL`, this function does nothing.
///
/// @param descriptor pointer to an existing descriptor, or `NULL`
///
/// @returns The status code of this operation. If the status is not
///          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn aml_descriptor_free(
    descriptor: *mut aml_descriptor_t,
) -> aml_status_t {
    catch_unwind(|| {
        if !descriptor.is_null() {
            std::mem::drop(Box::from_raw(descriptor));
        }

        Ok(())
    })
}


/// Get the sparse `labels` for the given `descriptor`. After a sucessful call
/// to this function, `labels.values` contains a pointer to memory inside the
/// `descriptor` which is invalidated when the descriptor is freed with
/// `aml_descriptor_free` or the set of sparse labels is modified by calling one
/// of the `aml_descriptor_sparse_to_XXX` function.
///
/// @param descriptor pointer to an existing descriptor
/// @param labels pointer to be filled with the sparse labels of the descriptor
///
/// @returns The status code of this operation. If the status is not
///          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn aml_descriptor_sparse_labels(
    descriptor: *const aml_descriptor_t,
    labels: *mut aml_labels_t,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, labels);

        *labels = (*descriptor).sparse().try_into()?;
        Ok(())
    })
}


/// Get a pointer to the `index`-th block in this descriptor.
///
/// The block memory is still managed by the descriptor, this block should not
/// be freed. The block is invalidated when the descriptor is freed with
/// `aml_descriptor_free` or the set of sparse labels is modified by calling one
/// of the `aml_descriptor_sparse_to_XXX` function.
///
/// @param descriptor pointer to an existing descriptor
/// @param block pointer to be filled with a block
/// @param index index of the block to get
///
/// @returns The status code of this operation. If the status is not
///          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
///          error message.
#[no_mangle]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern fn aml_descriptor_block_by_id(
    descriptor: *const aml_descriptor_t,
    block: *mut *const aml_block_t,
    index: u64,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, block);

        (*block) = (&(*descriptor).blocks()[index as usize] as *const Block).cast();

        Ok(())
    })
}


/// Get a pointer to the `block` in this `descriptor` corresponding to the given
/// `selection`. The `selection` should have the same names/variables as the
/// sparse labels for this descriptor, and only one entry, describing the
/// requested block.
///
/// The block memory is still managed by the descriptor, this block should not
/// be freed. The block is invalidated when the descriptor is freed with
/// `aml_descriptor_free` or the set of sparse labels is modified by calling one
/// of the `aml_descriptor_sparse_to_XXX` function.
///
/// @param descriptor pointer to an existing descriptor
/// @param block pointer to be filled with a block
/// @param selection labels with a single entry describing which block is requested
///
/// @returns The status code of this operation. If the status is not
///          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn aml_descriptor_block_selection(
    descriptor: *const aml_descriptor_t,
    block: *mut *const aml_block_t,
    selection: aml_labels_t,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, block);

        let selection = Labels::try_from(&selection)?;
        let rust_block = (*descriptor).block(&selection)?;
        (*block) = (rust_block as *const Block).cast();

        Ok(())
    })
}


/// Move the given variables from the sparse labels to the feature labels of the
/// blocks.
///
/// The current blocks will be merged together according to the sparse labels
/// remaining after removing `variables`. The resulting merged blocks will have
/// `variables` as the first feature variables, followed by the current
/// features. The new sample labels will contains all of the merged blocks
/// sample labels, re-ordered to keep them lexicographically sorted.
///
/// `variables` must be an array of `variables_count` NULL-terminated strings,
/// encoded as UTF-8.
///
/// @param descriptor pointer to an existing descriptor
/// @param variables name of the sparse variables to move to the features
/// @param variables_count number of entries in the `variables` array
///
/// @returns The status code of this operation. If the status is not
///          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
///          error message.
#[no_mangle]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern fn aml_descriptor_sparse_to_features(
    descriptor: *mut aml_descriptor_t,
    variables: *const *const c_char,
    variables_count: u64,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, variables);

        let mut rust_variables = Vec::new();
        for &variable in std::slice::from_raw_parts(variables, variables_count as usize) {
            check_pointers!(variable);
            let variable = CStr::from_ptr(variable).to_str().expect("invalid utf8");
            rust_variables.push(variable);
        }

        (*descriptor).sparse_to_features(&rust_variables)?;

        Ok(())
    })
}


/// Move the given variables from the component labels to the feature labels for
/// each block in this descriptor.
///
/// `variables` must be an array of `variables_count` NULL-terminated strings,
/// encoded as UTF-8.
///
/// @param descriptor pointer to an existing descriptor
/// @param variables name of the sparse variables to move to the features
/// @param variables_count number of entries in the `variables` array
///
/// @returns The status code of this operation. If the status is not
///          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
///          error message.
#[no_mangle]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern fn aml_descriptor_components_to_features(
    descriptor: *mut aml_descriptor_t,
    variables: *const *const c_char,
    variables_count: u64,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, variables);

        let mut rust_variables = Vec::new();
        for &variable in std::slice::from_raw_parts(variables, variables_count as usize) {
            check_pointers!(variable);
            let variable = CStr::from_ptr(variable).to_str().expect("invalid utf8");
            rust_variables.push(variable);
        }

        (*descriptor).components_to_features(&rust_variables)?;

        Ok(())
    })
}

/// Move the given variables from the sparse labels to the sample labels of the
/// blocks.
///
/// The current blocks will be merged together according to the sparse
/// labels remaining after removing `variables`. The resulting merged
/// blocks will have `variables` as the last sample variables, preceded by
/// the current samples.
///
/// `variables` must be an array of `variables_count` NULL-terminated strings,
/// encoded as UTF-8.
///
/// @param descriptor pointer to an existing descriptor
/// @param variables name of the sparse variables to move to the samples
/// @param variables_count number of entries in the `variables` array
///
/// @returns The status code of this operation. If the status is not
///          `AML_SUCCESS`, you can use `aml_last_error()` to get the full
///          error message.
#[no_mangle]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern fn aml_descriptor_sparse_to_samples(
    descriptor: *mut aml_descriptor_t,
    variables: *const *const c_char,
    variables_count: u64,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, variables);

        let mut rust_variables = Vec::new();
        for &variable in std::slice::from_raw_parts(variables, variables_count as usize) {
            check_pointers!(variable);
            let variable = CStr::from_ptr(variable).to_str().expect("invalid utf8");
            rust_variables.push(variable);
        }

        (*descriptor).sparse_to_samples(&rust_variables)?;

        Ok(())
    })
}
