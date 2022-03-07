use std::sync::Arc;
use std::os::raw::c_char;
use std::ffi::CStr;
use std::convert::TryFrom;

use crate::{Block, Labels, LabelValue, Error, aml_array_t};

use super::labels::{aml_labels_t, aml_label_kind};

use super::{catch_unwind, aml_status_t};

/// Opaque type representing a `Block`.
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

#[no_mangle]
pub unsafe extern fn aml_block(
    data: aml_array_t,
    samples: aml_labels_t,
    symmetric: aml_labels_t,
    features: aml_labels_t,
) -> *mut aml_block_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        let samples = Labels::try_from(samples)?;
        let symmetric = Labels::try_from(symmetric)?;
        let features = Labels::try_from(features)?;

        let block = Block::new(data, samples, Arc::new(symmetric), Arc::new(features))?;
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
            aml_label_kind::AML_SYMMETRIC_LABELS => basic_block.symmetric(),
            aml_label_kind::AML_FEATURE_LABELS => basic_block.features(),
        };

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
        // TODO: do we need `(*data).destroy = None` here ?

        Ok(())
    })
}

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
