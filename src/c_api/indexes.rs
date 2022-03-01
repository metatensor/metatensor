use std::os::raw::c_char;
use std::ffi::CStr;

use crate::{IndexValue, Indexes, IndexesBuilder, Error};

#[repr(C)]
#[allow(non_camel_case_types)]
/// The different kinds of indexes that can exist on a `aml_descriptor_t`
pub enum aml_indexes_kind {
    /// The samples index, describing different samples in the representation
    AML_INDEXES_SAMPLES = 0,
    /// TODO
    AML_INDEXES_SYMMETRIC = 1,
    /// The feature index, describing the features of the representation
    AML_INDEXES_FEATURES = 2,
}

/// Indexes representing metadata associated with either samples or features in
/// a given descriptor.
#[repr(C)]
pub struct aml_indexes_t {
    /// Names of the variables composing this set of indexes. There are `size`
    /// elements in this array, each being a NULL terminated string.
    pub names: *const *const c_char,
    /// Pointer to the first element of a 2D row-major array of 32-bit signed
    /// integer containing the values taken by the different variables in
    /// `names`. Each row has `size` elements, and there are `count` rows in
    /// total.
    pub values: *const i32,
    /// Number of variables/size of a single entry in the set of indexes
    pub size: usize,
    /// Number entries in the set of indexes
    pub count: usize,
}

impl TryFrom<aml_indexes_t> for Indexes {
    type Error = Error;

    fn try_from(indexes: aml_indexes_t) -> Result<Indexes, Self::Error> {
        if indexes.names.is_null() || indexes.values.is_null() {
            todo!()
        }

        let mut names = Vec::new();
        unsafe {
            for i in 0..indexes.size {
                let name = CStr::from_ptr(*(indexes.names.add(i)));
                names.push(name.to_str().expect("invalid UTF8 name"));
            }
        }

        let mut builder = IndexesBuilder::new(names);

        unsafe {
            let slice = std::slice::from_raw_parts(indexes.values.cast::<IndexValue>(), indexes.count * indexes.size);
            if !slice.is_empty() {
                for chunk in slice.chunks_exact(indexes.size) {
                    builder.add(chunk.to_vec());
                }
            }
        }

        return Ok(builder.finish());
    }
}
