use std::ffi::{CString, CStr};
use std::iter::FusedIterator;

use crate::c_api::eqs_block_t;
use crate::{ArrayRefMut, Labels};

use super::TensorBlockRef;
use super::block_ref::{block_array, block_metadata};

/// Mutable reference to a [`TensorBlock`](crate::TensorBlock)
#[derive(Debug)]
#[repr(transparent)]
pub struct TensorBlockRefMut<'a> {
    data: TensorBlockRef<'a>,
}

impl<'a> std::ops::Deref for TensorBlockRefMut<'a> {
    type Target = TensorBlockRef<'a>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a> std::ops::DerefMut for TensorBlockRefMut<'a> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// Mutable version of [`BasicBlock`](crate::BasicBlock)
#[derive(Debug)]
pub struct BasicBlockMut<'a> {
    /// Mutable reference to the data array
    pub data: ArrayRefMut<'a>,
    /// Labels describing the samples, i.e. the first dimension of the array
    ///
    /// If you need to get a reference to the samples with lifetime 'a, use the
    /// [`BasicBlock::samples_ref`](crate::BasicBlock::samples_ref) function.
    pub samples: Labels,
    /// Labels describing the components, i.e. the middle dimension of the array
    ///
    /// This can be empty if we don't have any symmetry or gradient components
    /// (and the data array is 2 dimensional)
    pub components: Vec<Labels>,
    /// Labels describing the properties, i.e. the last dimension of the array
    pub properties: Labels,
}

impl<'a> TensorBlockRefMut<'a> {
    /// Create a new `TensorBlockRefMut` from the given raw `eqs_block_t`
    ///
    /// This is a **VERY** unsafe function, creating a lifetime out of thin air,
    /// and allowing mutable access to the `eqs_block_t`. Make sure the lifetime
    /// is actually constrained by the lifetime of the owner of this
    /// `eqs_block_t`; and that the owner is mutably borrowed by this
    /// `TensorBlockRefMut`.
    pub(crate) unsafe fn from_raw(ptr: *mut eqs_block_t) -> TensorBlockRefMut<'a> {
        TensorBlockRefMut { data: TensorBlockRef::from_raw(ptr) }
    }

    /// Get the underlying (mutable) raw pointer
    pub(super) fn as_mut_ptr(&mut self) -> *mut eqs_block_t {
        self.as_ptr() as *mut _
    }

    /// Get a mutable reference to the values data and metadata in this block
    #[inline]
    pub fn values_mut(&mut self) -> BasicBlockMut<'_> {
        let values = unsafe { CStr::from_bytes_with_nul_unchecked(b"values\0") };
        let array = block_array(self.as_mut_ptr(), values).expect("failed to get values");

        let shape_len = array.shape().expect("failed to get the data shape").len();
        let (samples, components, properties) = block_metadata(self.as_ptr(), shape_len, values);

        BasicBlockMut {
            data: unsafe { ArrayRefMut::new(array) },
            samples,
            components,
            properties,
        }
    }

    /// Get a mutable reference to the data and metadata for the gradient with
    /// respect to the given parameter in this block, if it exists.
    #[inline]
    pub fn gradient_mut(&mut self, parameter: &str) -> Option<BasicBlockMut<'_>> {
        let parameter = CString::new(parameter).expect("invalid C string");
        let array = block_array(self.as_mut_ptr(), &parameter);

        if let Some(array) = array {
            let shape_len = array.shape().expect("failed to get the data shape").len();
            let (samples, components, properties) = block_metadata(self.as_ptr(), shape_len, &parameter);

            Some(BasicBlockMut {
                data: unsafe { ArrayRefMut::new(array) },
                samples,
                components,
                properties,
            })
        } else {
            None
        }
    }

    /// Get an iterator over parameter/[`BasicBlockMut`] pairs for all gradients
    /// in this block
    #[inline]
    pub fn gradients_mut(&mut self) -> GradientsMutIter<'_> {
        GradientsMutIter {
            parameters: self.gradient_list().into_iter(),
            block: self.as_ptr() as *mut eqs_block_t,
        }
    }
}

/// Iterator over parameter/[`BasicBlockMut`] pairs for all gradients in a
/// [`TensorBlockRefMut`]
pub struct GradientsMutIter<'a> {
    parameters: std::vec::IntoIter<&'a str>,
    block: *mut eqs_block_t,
}

impl<'a> Iterator for GradientsMutIter<'a> {
    type Item = (&'a str, BasicBlockMut<'a>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.parameters.next().map(|parameter| {
            let parameter_c = CString::new(parameter).expect("invalid C string");
            let array = block_array(self.block, &parameter_c).expect("missing gradient");

            let shape_len = array.shape().expect("failed to get the data shape").len();
            let (samples, components, properties) = block_metadata(self.block, shape_len, &parameter_c);

            let block = BasicBlockMut {
                data: unsafe { ArrayRefMut::new(array) },
                samples,
                components,
                properties,
            };

            return (parameter, block);
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.parameters.size_hint()
    }
}

impl<'a> ExactSizeIterator for GradientsMutIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.parameters.len()
    }
}

impl<'a> FusedIterator for GradientsMutIter<'a> {}

#[cfg(test)]
mod tests {
    use crate::c_api::eqs_block_t;
    use super::*;

    #[test]
    fn check_repr() {
        // we are transmuting `TensorBlock` to `*const eqs_block_t` in TensorMap::new,
        // this is only legal because TensorBlock == TensorBlockRefMut == TensorBlockRef == *const eqs_block_t
        assert_eq!(std::mem::size_of::<TensorBlockRefMut>(), std::mem::size_of::<*const eqs_block_t>());
        assert_eq!(std::mem::align_of::<TensorBlockRefMut>(), std::mem::align_of::<*const eqs_block_t>());
    }
}
