use std::os::raw::c_char;
use std::ffi::CStr;
use std::convert::TryFrom;
use std::collections::BTreeSet;

use crate::{Descriptor, Labels, LabelValue, Block, Error};

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

#[no_mangle]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern fn aml_descriptor(
    sparse_labels: aml_labels_t,
    blocks: *mut *mut aml_block_t,
    blocks_count: u64,
) -> *mut aml_descriptor_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        let sparse_labels = Labels::try_from(sparse_labels)?;

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

        let descriptor = Descriptor::new(sparse_labels, blocks_vec)?;
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

#[no_mangle]
pub unsafe extern fn aml_descriptor_sparse_labels(
    descriptor: *const aml_descriptor_t,
    labels: *mut aml_labels_t,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, labels);

        let rust_labels = (*descriptor).sparse();

        (*labels).size = rust_labels.size();
        (*labels).count = rust_labels.count();

        if rust_labels.count() == 0 || rust_labels.size() == 0 {
            (*labels).values = std::ptr::null();
        } else {
            (*labels).values = (&rust_labels[0][0] as *const LabelValue).cast();
        }

        if rust_labels.size() == 0 {
            (*labels).names = std::ptr::null();
        } else {
            (*labels).names = rust_labels.c_names().as_ptr().cast();
        }
        Ok(())
    })
}

#[no_mangle]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern fn aml_descriptor_block_by_id(
    descriptor: *const aml_descriptor_t,
    block: *mut *const aml_block_t,
    id: u64,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, block);

        (*block) = (&(*descriptor).blocks()[id as usize] as *const Block).cast();

        Ok(())
    })
}


#[no_mangle]
pub unsafe extern fn aml_descriptor_block_selection(
    descriptor: *const aml_descriptor_t,
    block: *mut *const aml_block_t,
    selection: aml_labels_t,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor, block);

        let selection = Labels::try_from(selection)?;
        let rust_block = (*descriptor).block(&selection)?;
        (*block) = (rust_block as *const Block).cast();

        Ok(())
    })
}


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

        (*descriptor).sparse_to_features(rust_variables)?;

        Ok(())
    })
}


#[no_mangle]
pub unsafe extern fn aml_descriptor_components_to_features(
    descriptor: *mut aml_descriptor_t,
) -> aml_status_t {
    catch_unwind(|| {
        check_pointers!(descriptor);

        (*descriptor).components_to_features()?;

        Ok(())
    })
}


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

        (*descriptor).sparse_to_samples(rust_variables)?;

        Ok(())
    })
}
