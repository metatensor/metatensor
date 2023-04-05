use std::ffi::{CString, CStr};
use std::iter::FusedIterator;

use crate::c_api::{eqs_block_t, eqs_array_t, EQS_INVALID_PARAMETER_ERROR};
use crate::ArrayRefMut;

use super::TensorBlockRef;

/// Mutable reference to a [`TensorBlock`](crate::TensorBlock)
#[derive(Debug)]
#[repr(transparent)]
pub struct TensorBlockRefMut<'a> {
    data: TensorBlockRef<'a>,
}

/// Get a gradient from this block
fn block_gradient(block: *mut eqs_block_t, parameter: &CStr) -> Option<*mut eqs_block_t> {
    let mut gradient_block = std::ptr::null_mut();
    let status = unsafe { crate::c_api::eqs_block_gradient(
            block,
            parameter.as_ptr(),
            &mut gradient_block
        )
    };

    match crate::errors::check_status(status) {
        Ok(_) => Some(gradient_block),
        Err(error) => {
            if error.code == Some(EQS_INVALID_PARAMETER_ERROR) {
                // there is no array for this gradient
                None
            } else {
                panic!("failed to get the gradient from a block: {:?}", error)
            }
        }
    }
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
        self.data.as_ptr() as *mut _
    }

    /// Get a non mutable reference to this block
    pub fn as_ref(&self) -> TensorBlockRef<'_> {
        unsafe {
            TensorBlockRef::from_raw(self.data.as_ptr())
        }
    }

    /// Get a mutable reference to the values in this block
    #[inline]
    pub fn values_mut(&mut self) -> ArrayRefMut<'_> {
        let mut array = eqs_array_t::null();
        unsafe {
            crate::errors::check_status(crate::c_api::eqs_block_data(
                self.as_mut_ptr(),
                &mut array
            )).expect("failed to get the array for a block");
        };

        // SAFETY: we are returning an `ArrayRefMut` mutably borrowing from `self`
        unsafe { ArrayRefMut::new(array) }
    }

    /// Get a mutable reference to the data and metadata for the gradient with
    /// respect to the given parameter in this block, if it exists.
    #[inline]
    pub fn gradient_mut(&mut self, parameter: &str) -> Option<TensorBlockRefMut<'_>> {
        let parameter = CString::new(parameter).expect("invalid C string");

        block_gradient(self.as_mut_ptr(), &parameter)
            .map(|gradient_block| {
                // SAFETY: we are returning an `TensorBlockRefMut` mutably
                // borrowing from `self`
                unsafe { TensorBlockRefMut::from_raw(gradient_block) }
            })
    }

    /// Get an iterator over parameter/[`TensorBlockRefMut`] pairs for all gradients
    /// in this block
    #[inline]
    pub fn gradients_mut(&mut self) -> GradientsMutIter<'_> {
        GradientsMutIter {
            parameters: self.as_ref().gradient_list().into_iter(),
            block: self.data.as_ptr() as *mut _,
        }
    }
}

/// Iterator over parameter/[`TensorBlockRefMut`] pairs for all gradients in a
/// [`TensorBlockRefMut`]
pub struct GradientsMutIter<'a> {
    parameters: std::vec::IntoIter<&'a str>,
    block: *mut eqs_block_t,
}

impl<'a> Iterator for GradientsMutIter<'a> {
    type Item = (&'a str, TensorBlockRefMut<'a>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.parameters.next().map(|parameter| {
            let parameter_c = CString::new(parameter).expect("invalid C string");
            let block = block_gradient(self.block, &parameter_c).expect("missing gradient");

            // SAFETY: all blocks are disjoint, and we are only returning a
            // mutable reference to each once. The reference lifetime is
            // constrained by the lifetime of the parent TensorBlockRefMut
            let block = unsafe { TensorBlockRefMut::from_raw(block) };
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
