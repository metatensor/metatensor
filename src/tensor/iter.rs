use crate::{TensorBlock, TensorBlockRefMut};
use crate::{LabelValue};

use super::TensorMap;


impl TensorMap {
    /// Get an iterator over the keys and associated blocks
    pub fn iter(&self) -> Iter {
        Iter {
            inner: self.keys.iter().zip(&self.blocks)
        }
    }

    /// Get an iterator over the keys and associated blocks, with read-write
    /// access to the blocks
    pub fn iter_mut(&mut self) -> IterMut {
        IterMut {
            inner: self.keys.iter().zip(&mut self.blocks)
        }
    }

    /// Get a parallel iterator over the keys and associated blocks
    #[cfg(feature = "rayon")]
    pub fn par_iter(&self) -> ParIter {
        use rayon::prelude::*;
        ParIter {
            inner: self.keys.par_iter().zip_eq(&self.blocks)
        }
    }

    /// Get a parallel iterator over the keys and associated blocks, with
    /// read-write access to the blocks
    #[cfg(feature = "rayon")]
    pub fn par_iter_mut(&mut self) -> ParIterMut {
        use rayon::prelude::*;
        ParIterMut {
            inner: self.keys.par_iter().zip_eq(&mut self.blocks)
        }
    }
}

/******************************************************************************/

/// Iterator over key/block pairs in a `TensorMap`
pub struct Iter<'a> {
    inner: std::iter::Zip<crate::labels::Iter<'a>, std::slice::Iter<'a, TensorBlock>>
}

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a [LabelValue], &'a TensorBlock);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'a> ExactSizeIterator for Iter<'a> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/******************************************************************************/

/// Iterator over key/block pairs in a `TensorMap`, with mutable access to the
/// blocks
pub struct IterMut<'a> {
    inner: std::iter::Zip<crate::labels::Iter<'a>, std::slice::IterMut<'a, TensorBlock>>
}

impl<'a> Iterator for IterMut<'a> {
    type Item = (&'a [LabelValue], TensorBlockRefMut<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((key, block)) = self.inner.next() {
            Some((key, block.as_mut()))
        } else {
            None
        }

    }
}

impl<'a> ExactSizeIterator for IterMut<'a> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/******************************************************************************/

/// Parallel iterator over key/block pairs in a `TensorMap`
#[cfg(feature = "rayon")]
pub struct ParIter<'a> {
    inner: rayon::iter::ZipEq<crate::labels::ParIter<'a>, rayon::slice::Iter<'a, TensorBlock>>,
}

#[cfg(feature = "rayon")]
impl<'a> rayon::iter::ParallelIterator for ParIter<'a> {
    type Item = (&'a [LabelValue], &'a TensorBlock);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item> {
        self.inner.drive_unindexed(consumer)
    }
}

#[cfg(feature = "rayon")]
impl<'a> rayon::iter::IndexedParallelIterator for ParIter<'a> {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        self.inner.drive(consumer)
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        self.inner.with_producer(callback)
    }
}

/******************************************************************************/

/// Parallel iterator over key/block pairs in a `TensorMap`, with mutable access
/// to the blocks
#[cfg(feature = "rayon")]
pub struct ParIterMut<'a> {
    inner: rayon::iter::ZipEq<crate::labels::ParIter<'a>, rayon::slice::IterMut<'a, TensorBlock>>,
}

#[cfg(feature = "rayon")]
impl<'a> rayon::iter::ParallelIterator for ParIterMut<'a> {
    type Item = (&'a [LabelValue], TensorBlockRefMut<'a>);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item> {
        self.inner.map(|(k, b)| (k, b.as_mut())).drive_unindexed(consumer)
    }
}

#[cfg(feature = "rayon")]
impl<'a> rayon::iter::IndexedParallelIterator for ParIterMut<'a> {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        use rayon::prelude::*;
        self.inner.map(|(k, b)| (k, b.as_mut())).drive(consumer)
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        use rayon::prelude::*;
        self.inner.map(|(k, b)| (k, b.as_mut())).with_producer(callback)
    }
}
