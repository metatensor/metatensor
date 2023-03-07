use crate::c_api::eqs_block_t;
use crate::errors::check_status;
use crate::{Array, Labels, Error, TensorBlockRef};

use super::TensorBlockRefMut;

/// A single block, containing both values & optionally gradients of these
/// values w.r.t. any relevant quantity.
#[derive(Debug)]
#[repr(transparent)]
pub struct TensorBlock {
    pub(super) data: TensorBlockRefMut<'static>
}

impl std::ops::Drop for TensorBlock {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        unsafe {
            crate::c_api::eqs_block_free(self.as_ref_mut().as_mut_ptr());
        }
    }
}

impl TensorBlock {
    /// Create a new `TensorBlock` from a raw pointer.
    ///
    /// This function takes ownership of the pointer, and will call
    /// `eqs_block_free` on it when the `TensorBlock` goes out of scope.
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and point to a owned block, not a reference
    /// to a block from inside a [`TensorMap`](crate::TensorMap).
    pub(crate) unsafe fn from_raw(ptr: *mut eqs_block_t) -> TensorBlock {
        TensorBlock { data: TensorBlockRefMut::from_raw(ptr) }
    }

    /// Get a reference to this owned block, with a lifetime of `'self`.
    pub fn as_ref(&self) -> TensorBlockRef<'_> {
        // This is not implemented with `std::ops::Deref`, because the lifetime
        // of the resulting TensorBlockRef should not be `'static` but `'self`.
        unsafe {
            TensorBlockRef::from_raw(self.data.as_ref().as_ptr())
        }
    }

    /// Get a mutable reference to this block, with a lifetime of `'self`.
    pub fn as_ref_mut(&mut self) -> TensorBlockRefMut<'_> {
        // This is not implemented with `std::ops::DerefMut`, because the lifetime
        // of the resulting TensorBlockRefMut should not be `'static` but `'self`.
        unsafe {
            TensorBlockRefMut::from_raw(self.data.as_mut_ptr())
        }
    }

    /// Create a new [`TensorBlock`] containing the given data, described by the
    /// `samples`, `components`, and `properties` labels. The block is
    /// initialized without any gradients.
    #[allow(clippy::needless_pass_by_value)]
    #[inline]
    pub fn new(
        data: impl Array,
        samples: Labels,
        components: &[Labels],
        properties: Labels
    ) -> Result<TensorBlock, Error> {
        let mut c_components = Vec::new();
        for component in components {
            c_components.push(component.as_eqs_labels_t());
        }

        let ptr = unsafe {
            crate::c_api::eqs_block(
                (Box::new(data) as Box<dyn Array>).into(),
                samples.as_eqs_labels_t(),
                c_components.as_ptr(),
                c_components.len(),
                properties.as_eqs_labels_t(),
            )
        };

        crate::errors::check_ptr(ptr)?;

        return Ok(unsafe { TensorBlock::from_raw(ptr) });
    }

    /// Add a gradient with respect to `parameter` to this block.
    ///
    /// The gradient `data` is given as an array, and the samples and components
    /// labels must be provided. The property labels are assumed to match the
    /// ones of the values in this block.
    ///
    /// The components labels must contain at least the same entries as the
    /// value components labels, and can prepend other components labels.
    #[allow(clippy::needless_pass_by_value)]
    #[inline]
    pub fn add_gradient(
        &mut self,
        parameter: &str,
        data: impl Array,
        samples: Labels,
        components: &[Labels],
    ) -> Result<(), Error> {
        let mut parameter = parameter.to_owned().into_bytes();
        parameter.push(b'\0');

        let c_components = components.iter().map(|c| c.as_eqs_labels_t()).collect::<Vec<_>>();

        let mut data = (Box::new(data) as Box<dyn Array>).into();
        unsafe {
            check_status(crate::c_api::eqs_block_add_gradient(
                self.as_ref_mut().as_mut_ptr(),
                parameter.as_ptr().cast(),
                data,
                samples.as_eqs_labels_t(),
                c_components.as_ptr(),
                c_components.len(),
            ))?;

            // do not drop the data here, it will be dropped from inside the
            // block
            data.destroy = None;
        }

        return Ok(());
    }
}


#[cfg(test)]
mod tests {
    use crate::c_api::eqs_block_t;
    use super::*;

    #[test]
    fn check_repr() {
        // we are transmuting `TensorBlock` to `*const eqs_block_t` in TensorMap::new,
        // this is only legal because TensorBlock == TensorBlockRefMut == TensorBlockRef == *const eqs_block_t
        assert_eq!(std::mem::size_of::<TensorBlock>(), std::mem::size_of::<*const eqs_block_t>());
        assert_eq!(std::mem::align_of::<TensorBlock>(), std::mem::align_of::<*const eqs_block_t>());
    }
}
