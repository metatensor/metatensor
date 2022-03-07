use std::os::raw::c_char;
use std::ffi::CStr;

use crate::{LabelValue, Labels, LabelsBuilder, Error};

#[repr(C)]
#[allow(non_camel_case_types)]
/// The different kinds of labels that can exist on a `aml_descriptor_t`
pub enum aml_label_kind {
    /// The sample labels, describing different samples in the representation
    AML_SAMPLE_LABELS = 0,
    /// TODO
    AML_SYMMETRIC_LABELS = 1,
    /// The feature labels, describing the features of the representation
    AML_FEATURE_LABELS = 2,
}

/// Labels representing metadata associated with either samples or features in
/// a given descriptor.
#[repr(C)]
pub struct aml_labels_t {
    /// Names of the variables composing this set of labels. There are `size`
    /// elements in this array, each being a NULL terminated string.
    pub names: *const *const c_char,
    /// Pointer to the first element of a 2D row-major array of 32-bit signed
    /// integer containing the values taken by the different variables in
    /// `names`. Each row has `size` elements, and there are `count` rows in
    /// total.
    pub values: *const i32,
    /// Number of variables/size of a single entry in the set of labels
    pub size: usize,
    /// Number entries in the set of labels
    pub count: usize,
}

impl std::convert::TryFrom<aml_labels_t> for Labels {
    type Error = Error;

    fn try_from(labels: aml_labels_t) -> Result<Labels, Self::Error> {
        if labels.names.is_null() || labels.values.is_null() {
            todo!()
        }

        let mut names = Vec::new();
        unsafe {
            for i in 0..labels.size {
                let name = CStr::from_ptr(*(labels.names.add(i)));
                names.push(name.to_str().expect("invalid UTF8 name"));
            }
        }

        let mut builder = LabelsBuilder::new(names);

        unsafe {
            let slice = std::slice::from_raw_parts(labels.values.cast::<LabelValue>(), labels.count * labels.size);
            if !slice.is_empty() {
                for chunk in slice.chunks_exact(labels.size) {
                    builder.add(chunk.to_vec());
                }
            }
        }

        return Ok(builder.finish());
    }
}
