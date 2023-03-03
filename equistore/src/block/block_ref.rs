use std::ffi::{CStr, CString};
use std::iter::FusedIterator;

use crate::c_api::{eqs_block_t, eqs_array_t, eqs_labels_t};
use crate::c_api::EQS_INVALID_PARAMETER_ERROR;

use crate::errors::check_status;
use crate::{ArrayRef, Labels, Error};

use super::TensorBlock;

/// Reference to a [`TensorBlock`]
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct TensorBlockRef<'a> {
    ptr: *const eqs_block_t,
    marker: std::marker::PhantomData<&'a eqs_block_t>,
}

// SAFETY: Send is fine since we can free a TensorBlock from any thread
unsafe impl<'a> Send for TensorBlockRef<'a> {}
// SAFETY: Sync is fine since there is no internal mutability in TensorBlock
unsafe impl<'a> Sync for TensorBlockRef<'a> {}

/// Single data array with the corresponding metadata inside a [`TensorBlock`]
///
/// A basic block contains a n-dimensional array, and n sets of labels (one for
/// each dimension). The first dimension labels are called the `samples`, the
/// last dimension labels are the `properties`, and everything else is in the
/// `components`
#[derive(Debug, Clone)]
pub struct BasicBlock<'a> {
    /// Reference to the data array
    pub data: ArrayRef<'a>,
    /// Labels describing the samples, i.e. the first dimension of the array
    ///
    /// If you need to get a reference to the samples with lifetime 'a, use the
    /// [`BasicBlock::samples_ref`] function.
    pub samples: Labels,
    /// Labels describing the components, i.e. the middle dimension of the array
    ///
    /// This can be empty if we don't have any symmetry or gradient components
    /// (and the data array is 2 dimensional)
    pub components: Vec<Labels>,
    /// Labels describing the properties, i.e. the last dimension of the array
    pub properties: Labels,
}

impl<'a> BasicBlock<'a> {
    /// Get a reference with lifetime `'a` to the samples of this `BasicBlock`
    pub fn samples_ref(&self) -> &'a Labels {
        let ptr = &self.samples as *const _;
        // SAFETY: we are only handing out shared references, which can not
        // outlive `self`
        unsafe {
            return &*ptr;
        }
    }
}

impl<'a> TensorBlockRef<'a> {
    /// Create a new `TensorBlockRef` from the given raw `eqs_block_t`
    ///
    /// This is a **VERY** unsafe function, creating a lifetime out of thin air.
    /// Make sure the lifetime is actually constrained by the lifetime of the
    /// owner of this `eqs_block_t`.
    pub(crate) unsafe fn from_raw(ptr: *const eqs_block_t) -> TensorBlockRef<'a> {
        assert!(!ptr.is_null(), "pointer to eqs_block_t should not be NULL");

        TensorBlockRef {
            ptr: ptr,
            marker: std::marker::PhantomData,
        }
    }

    /// Get the underlying raw pointer
    pub(super) fn as_ptr(&self) -> *const eqs_block_t {
        self.ptr
    }
}

fn get_block_label(block: *const eqs_block_t, dimension: usize, values_gradient: &CStr) -> Labels  {
    let mut labels = eqs_labels_t::null();
    unsafe {
        check_status(crate::c_api::eqs_block_labels(
            block,
            values_gradient.as_ptr(),
            dimension,
            &mut labels,
        )).expect("failed to get labels");
    }
    return unsafe { Labels::from_raw(labels) };
}

/// Get the metadata for the values or one gradient, depending on
/// `values_gradient`. See `eqs_block_labels` for more information.
pub(super) fn block_metadata(
    block: *const eqs_block_t,
    shape_len: usize,
    values_gradient: &CStr,
) -> (Labels, Vec<Labels>, Labels) {
    let samples = get_block_label(block, 0, values_gradient);

    let mut components = Vec::new();
    for dimension in 1..(shape_len - 1) {
        components.push(get_block_label(block, dimension, values_gradient));
    }

    let properties = get_block_label(block, shape_len - 1, values_gradient);

    return (samples, components, properties);
}


/// Get the array associated with `values_gradient` in this block
pub(super) fn block_array(block: *mut eqs_block_t, values_gradient: &CStr) -> Option<eqs_array_t> {
    let mut array = eqs_array_t::null();
    let status = unsafe { crate::c_api::eqs_block_data(
        block,
        values_gradient.as_ptr(),
        &mut array
    )};

    match crate::errors::check_status(status) {
        Ok(_) => Some(array),
        Err(error) => {
            if error.code == Some(EQS_INVALID_PARAMETER_ERROR) {
                // there is no array for this gradient
                return None
            } else {
                panic!("failed to get the array for {:?}: {:?}", values_gradient, error)
            }
        }
    }
}

impl<'a> TensorBlockRef<'a> {
    /// Get the values data and metadata in this block

    // SAFETY: we can return a basic block with lifetime `'a` (instead of
    // `'self`) (which allows to get multiple references to the BasicBlock
    // simultaneously), because there is no way to also get a mutable reference
    // to the block at the same time.
    #[inline]
    pub fn values(&self) -> BasicBlock<'a> {
        let values = unsafe { CStr::from_bytes_with_nul_unchecked(b"values\0") };

        // the cast to mut pointer is fine since we are only returning a non-mut
        // ArrayRef below
        let array = block_array(self.as_ptr() as *mut _, values).expect("failed to get values");
        let shape_len = array.shape().expect("failed to get the data shape").len();

        let (samples, components, properties) = block_metadata(self.as_ptr(), shape_len, values);

        BasicBlock {
            data: unsafe { ArrayRef::from_raw(array) },
            samples,
            components,
            properties,
        }
    }

    /// Get the full list of gradients in this block

    // SAFETY: we can return strings with the `'a` lifetime (instead of
    // `'self`), because there is no way to also get a mutable reference
    // to the gradient parameters at the same time.
    #[inline]
    pub fn gradient_list(&self) -> Vec<&'a str> {
        let mut parameters_ptr = std::ptr::null();
        let mut parameters_count = 0;
        unsafe {
            check_status(crate::c_api::eqs_block_gradients_list(
                self.as_ptr(),
                &mut parameters_ptr,
                &mut parameters_count
            )).expect("failed to get gradient list");
        }

        unsafe {
            let parameters = std::slice::from_raw_parts(parameters_ptr, parameters_count);
            return parameters.iter()
                .map(|&ptr| CStr::from_ptr(ptr).to_str().unwrap())
                .collect();
        }
    }

    /// Get the data and metadata for the gradient with respect to the given
    /// parameter in this block, if it exists.

    // SAFETY: we can return a basic block with lifetime `'a` (instead of
    // `'self`) for the same reasons as in the `values` function.
    #[inline]
    pub fn gradient(&self, parameter: &str) -> Option<BasicBlock<'a>> {
        let parameter = CString::new(parameter).expect("invalid C string");

        // the cast to mut pointer is fine since we are only returning a non-mut
        // ArrayRef below
        if let Some(array) = block_array(self.as_ptr() as *mut _, &parameter) {
            let shape_len = array.shape().expect("failed to get the data shape").len();
            let (samples, components, properties) = block_metadata(self.as_ptr(), shape_len, &parameter);

            Some(BasicBlock {
                data: unsafe { ArrayRef::from_raw(array) },
                samples,
                components,
                properties,
            })
        } else {
            None
        }
    }

    /// Clone this block, cloning all the data and metadata contained inside.
    ///
    /// This can fail if the external data held inside an `eqs_array_t` can not
    /// be cloned.
    #[inline]
    pub fn try_clone(&self) -> Result<TensorBlock, Error> {
        let ptr = unsafe {
            crate::c_api::eqs_block_copy(self.as_ptr())
        };
        crate::errors::check_ptr(ptr)?;

        return Ok(unsafe { TensorBlock::from_raw(ptr) });
    }

    /// Get an iterator over parameter/[`BasicBlock`] pairs for all gradients in
    /// this block
    #[inline]
    pub fn gradients(&self) -> GradientsIter<'_> {
        GradientsIter {
            parameters: self.gradient_list().into_iter(),
            block: self.as_ptr(),
        }
    }
}

/// Iterator over parameter/[`BasicBlock`] pairs for all gradients in a
/// [`TensorBlockRef`]
pub struct GradientsIter<'a> {
    parameters: std::vec::IntoIter<&'a str>,
    block: *const eqs_block_t,
}

impl<'a> Iterator for GradientsIter<'a> {
    type Item = (&'a str, BasicBlock<'a>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.parameters.next().map(|parameter| {
            let parameter_c = CString::new(parameter).expect("invalid C string");

            // the cast to mut pointer is fine since we are only returning a
            // non-mut ArrayRef below
            let array = block_array(self.block as *mut _, &parameter_c).expect("missing gradient");

            let shape_len = array.shape().expect("failed to get the data shape").len();
            let (samples, components, properties) = block_metadata(self.block, shape_len, &parameter_c);

            let block = BasicBlock {
                data: unsafe { ArrayRef::from_raw(array) },
                samples,
                components,
                properties,
            };

            return (parameter, block);
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a> ExactSizeIterator for GradientsIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.parameters.len()
    }
}

impl<'a> FusedIterator for GradientsIter<'a> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_repr() {
        // we are transmuting `TensorBlock` to `*const eqs_block_t` in TensorMap::new,
        // this is only legal because TensorBlock == TensorBlockRefMut == TensorBlockRef == *const eqs_block_t
        assert_eq!(std::mem::size_of::<TensorBlockRef>(), std::mem::size_of::<*const eqs_block_t>());
        assert_eq!(std::mem::align_of::<TensorBlockRef>(), std::mem::align_of::<*const eqs_block_t>());
    }
}
