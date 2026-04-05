use std:: ffi::CStr;
use std::ffi::CString;
use std::collections::BTreeSet;
use std::iter::FusedIterator;

use smallvec::SmallVec;

use crate::c_api::mts_labels_t;
use crate::errors::{Error, check_status};

/// A single value inside a label.
///
/// This is represented as a 32-bit signed integer, with a couple of helper
/// integer
pub type LabelValue = i32;

/// A set of labels used to carry metadata associated with a tensor map.
///
/// This is similar to a list of named tuples, but stored as a 2D array of shape
/// `(labels.count(), labels.size())`, with a of set names associated with the
/// columns of this array. Each row/entry in this array is unique, and they are
/// often (but not always) sorted in  lexicographic order.
///
/// The main way to construct a new set of labels is to use a `LabelsBuilder`.
///
/// Labels are internally reference counted and immutable, so cloning a `Labels`
/// should be a cheap operation.
pub struct Labels {
    pub(crate) ptr: *mut mts_labels_t,
}

// Labels can be sent to other thread safely since mts_labels_t uses an
// `Arc<metatensor_core::Labels>`, so freeing them from another thread is fine
unsafe impl Send for Labels {}
// &Labels can be sent to other thread safely since the interior mutability
// (values array) uses OnceCell, which is Sync.
unsafe impl Sync for Labels {}

impl std::fmt::Debug for Labels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        pretty_print_labels(self, "", f)
    }
}

/// Helper function to print labels in a Debug mode
pub(crate) fn pretty_print_labels(
    labels: &Labels,
    offset: &str,
    f: &mut std::fmt::Formatter<'_>
) -> std::fmt::Result {
    let names = labels.names();

    writeln!(f, "Labels @ {:p} {{", labels.ptr)?;
    writeln!(f, "{}    {}", offset, names.join(", "))?;

    let widths = names.iter().map(|s| s.len()).collect::<Vec<_>>();
    for values in labels {
        write!(f, "{}    ", offset)?;
        for (value, width) in values.iter().zip(&widths) {
            write!(f, "{:^width$}  ", value, width=width)?;
        }
        writeln!(f)?;
    }

    writeln!(f, "{}}}", offset)
}

impl Clone for Labels {
    #[inline]
    fn clone(&self) -> Self {
        let ptr = unsafe { crate::c_api::mts_labels_clone(self.ptr) };
        assert!(!ptr.is_null(), "failed to clone Labels");
        Labels { ptr }
    }
}

impl std::ops::Drop for Labels {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        unsafe {
            crate::c_api::mts_labels_free(self.ptr);
        }
    }
}

impl Labels {
    /// Create a new set of Labels with the given names and values.
    ///
    /// This is a convenience function replacing the manual use of
    /// `LabelsBuilder`. If you need more flexibility or incremental `Labels`
    /// construction, use `LabelsBuilder`.
    ///
    /// # Panics
    ///
    /// If the set of names is not valid, or any of the value is duplicated
    #[inline]
    pub fn new<T, const N: usize>(names: [&str; N], values: &[[T; N]]) -> Labels
        where T: Copy + Into<LabelValue>
    {
        let mut builder = LabelsBuilder::new(names.to_vec());
        for entry in values {
            builder.add(entry);
        }
        return builder.finish();
    }

    /// Create a set of `Labels` with the given names, containing no entries.
    #[inline]
    pub fn empty(names: Vec<&str>) -> Labels {
        return LabelsBuilder::new(names).finish()
    }

    /// Create a set of `Labels` containing a single entry, to be used when
    /// there is no relevant information to store.
    #[inline]
    pub fn single() -> Labels {
        let mut builder = LabelsBuilder::new(vec!["_"]);
        builder.add(&[0]);
        return builder.finish();
    }

    /// Load `Labels` from the file at `path`
    ///
    /// This is a convenience function calling [`crate::io::load_labels`]
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Labels, Error> {
        return crate::io::load_labels(path);
    }

    /// Load a `TensorMap` from an in-memory buffer
    ///
    /// This is a convenience function calling [`crate::io::load_buffer`]
    pub fn load_buffer(buffer: &[u8]) -> Result<Labels, Error> {
        return crate::io::load_labels_buffer(buffer);
    }

    /// Save the given tensor to the file at `path`
    ///
    /// This is a convenience function calling [`crate::io::save`]
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), Error> {
        return crate::io::save_labels(path, self);
    }

    /// Save the given tensor to an in-memory buffer
    ///
    /// This is a convenience function calling [`crate::io::save_buffer`]
    pub fn save_buffer(&self, buffer: &mut Vec<u8>) -> Result<(), Error> {
        return crate::io::save_labels_buffer(self, buffer);
    }

    /// Get the number of entries/named values in a single label
    #[inline]
    pub fn size(&self) -> usize {
        self.names().len()
    }

    /// Get the names of the entries/columns in this set of labels
    #[inline]
    pub fn names(&self) -> Vec<&str> {
        let mut names_ptr = std::ptr::null();
        let mut count = 0;
        unsafe {
            check_status(crate::c_api::mts_labels_dimensions(self.ptr, &mut names_ptr, &mut count))
                .expect("failed to get labels dimensions");
        }

        if count == 0 {
            return Vec::new();
        }

        unsafe {
            let names = std::slice::from_raw_parts(names_ptr, count);
            return names.iter()
                        .map(|&ptr| CStr::from_ptr(ptr).to_str().expect("invalid UTF8"))
                        .collect();
        }
    }

    /// Get the total number of entries in this set of labels
    #[inline]
    pub fn count(&self) -> usize {
        let mut array = crate::c_api::mts_array_t::null();
        unsafe {
            check_status(crate::c_api::mts_labels_values(
                self.ptr, &mut array,
            )).expect("failed to get labels values array");
        }

        let mut shape_ptr = std::ptr::null();
        let mut shape_count: usize = 0;
        unsafe {
            let shape_fn = array.shape.expect("labels array must have shape callback");
            shape_fn(array.ptr, &mut shape_ptr, &mut shape_count);
            assert!(shape_count == 2, "labels array must be 2D");
            (*shape_ptr)
        }
    }

    /// Check if this set of Labels is empty (contains no entry)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Check whether the given `label` is part of this set of labels
    #[inline]
    pub fn contains(&self, label: &[LabelValue]) -> bool {
        return self.position(label).is_some();
    }

    /// Get the position (i.e. row index) of the given label in the full labels
    /// array, or None.
    #[inline]
    pub fn position(&self, value: &[LabelValue]) -> Option<usize> {
        assert!(value.len() == self.size(), "invalid size of index in Labels::position");

        let mut result = 0;
        unsafe {
            check_status(crate::c_api::mts_labels_position(
                self.ptr,
                value.as_ptr().cast(),
                value.len(),
                &mut result,
            )).expect("failed to check label position");
        }

        return result.try_into().ok();
    }

    /// Take the union of `self` with `other`.
    ///
    /// If requested, this function can also give the positions in the union
    /// where each entry of the input `Labels` ended up.
    ///
    /// If `first_mapping` (respectively `second_mapping`) is `Some`, it should
    /// contain a slice of length `self.count()` (respectively `other.count()`)
    /// that will be filled with the position of the entries in `self`
    /// (respectively `other`) in the union.
    #[inline]
    pub fn union(
        &self,
        other: &Labels,
        first_mapping: Option<&mut [i64]>,
        second_mapping: Option<&mut [i64]>,
    ) -> Result<Labels, Error> {
        let mut output: *mut mts_labels_t = std::ptr::null_mut();
        let (first_mapping, first_mapping_count) = if let Some(m) = first_mapping {
            (m.as_mut_ptr(), m.len())
        } else {
            (std::ptr::null_mut(), 0)
        };

        let (second_mapping, second_mapping_count) = if let Some(m) = second_mapping {
            (m.as_mut_ptr(), m.len())
        } else {
            (std::ptr::null_mut(), 0)
        };

        unsafe {
            check_status(crate::c_api::mts_labels_union(
                self.ptr,
                other.ptr,
                &mut output,
                first_mapping,
                first_mapping_count,
                second_mapping,
                second_mapping_count,
            ))?;

            return Ok(Labels::from_raw(output));
        }
    }

    /// Take the intersection of self with `other`.
    ///
    /// If requested, this function can also give the positions in the
    /// intersection where each entry of the input `Labels` ended up.
    ///
    /// If `first_mapping` (respectively `second_mapping`) is `Some`, it should
    /// contain a slice of length `self.count()` (respectively `other.count()`)
    /// that will be filled by with the position of the entries in `self`
    /// (respectively `other`) in the intersection. If an entry in `self` or
    /// `other` are not used in the intersection, the mapping for this entry
    /// will be set to `-1`.
    #[inline]
    pub fn intersection(
        &self,
        other: &Labels,
        first_mapping: Option<&mut [i64]>,
        second_mapping: Option<&mut [i64]>,
    ) -> Result<Labels, Error> {
        let mut output: *mut mts_labels_t = std::ptr::null_mut();
        let (first_mapping, first_mapping_count) = if let Some(m) = first_mapping {
            (m.as_mut_ptr(), m.len())
        } else {
            (std::ptr::null_mut(), 0)
        };

        let (second_mapping, second_mapping_count) = if let Some(m) = second_mapping {
            (m.as_mut_ptr(), m.len())
        } else {
            (std::ptr::null_mut(), 0)
        };

        unsafe {
            check_status(crate::c_api::mts_labels_intersection(
                self.ptr,
                other.ptr,
                &mut output,
                first_mapping,
                first_mapping_count,
                second_mapping,
                second_mapping_count,
            ))?;

            return Ok(Labels::from_raw(output));
        }
    }

    /// Take the set difference of `self` with `other`.
    ///
    /// If requested, this function can also give the positions in the
    /// difference where each entry of `self` ended up.
    ///
    /// If `mapping` is `Some`, it should contain a slice of length
    /// `self.count()` that will be filled by with the position of the entries
    /// in `self` in the difference. If an entry is not used in the difference,
    ///  the mapping for this entry will be set to `-1`.
    #[inline]
    pub fn difference(
        &self,
        other: &Labels,
        mapping: Option<&mut [i64]>,
    ) -> Result<Labels, Error> {
        let mut output: *mut mts_labels_t = std::ptr::null_mut();
        let (mapping, mapping_count) = if let Some(m) = mapping {
            (m.as_mut_ptr(), m.len())
        } else {
            (std::ptr::null_mut(), 0)
        };

        unsafe {
            check_status(crate::c_api::mts_labels_difference(
                self.ptr,
                other.ptr,
                &mut output,
                mapping,
                mapping_count,
            ))?;

            return Ok(Labels::from_raw(output));
        }
    }

    /// Iterate over the entries in this set of labels
    #[inline]
    pub fn iter(&self) -> LabelsIter<'_> {
        return LabelsIter {
            ptr: self.values().as_ptr(),
            cur: 0,
            len: self.count(),
            chunk_len: self.size(),
            phantom: std::marker::PhantomData,
        };
    }

    /// Iterate over the entries in this set of labels in parallel
    #[cfg(feature = "rayon")]
    #[inline]
    pub fn par_iter(&self) -> LabelsParIter<'_> {
        use rayon::prelude::*;
        return LabelsParIter {
            chunks: self.values().par_chunks_exact(self.size())
        };
    }

    /// Iterate over the entries in this set of labels as fixed-size arrays
    #[inline]
    pub fn iter_fixed_size<const N: usize>(&self) -> LabelsFixedSizeIter<'_, N> {
        assert!(N == self.size(),
            "wrong label size in `iter_fixed_size`: the entries contains {} element \
            but this function was called with size of {}",
            self.size(), N
        );

        return LabelsFixedSizeIter {
            values: self.values()
        };
    }

    /// Select entries in these `Labels` that match the `selection`.
    ///
    /// The selection's names must be a subset of the names of these labels.
    ///
    /// All entries in these `Labels` that match one of the entry in the
    /// `selection` for all the selection's dimension will be picked. Any entry
    /// in the `selection` but not in these `Labels` will be ignored.
    pub fn select(&self, selection: &Labels) -> Result<Vec<i64>, Error> {
        let mut selected = vec![-1; self.count()];
        let mut selected_count = selected.len();

        unsafe {
            check_status(crate::c_api::mts_labels_select(
                self.as_mts_labels_t(),
                selection.as_mts_labels_t(),
                selected.as_mut_ptr(),
                &mut selected_count
            ))?;
        }

        selected.resize(selected_count, 0);

        return Ok(selected);
    }

    pub(crate) fn values(&self) -> &[LabelValue] {
        let count = self.count();
        let size = self.size();
        if count == 0 || size == 0 {
            return &[]
        }

        // Get the values array, then extract CPU data via DLPack.
        // The DLPack tensor from LabelsValuesArray is zero-copy (points into
        // the array's own data), so this pointer is valid as long as the
        // Labels (and thus the mts_labels_t) is alive.
        let mut array = crate::c_api::mts_array_t::null();
        unsafe {
            check_status(crate::c_api::mts_labels_values(
                self.ptr, &mut array,
            )).expect("failed to get labels values array");
        }

        let cpu = dlpk::sys::DLDevice::cpu();
        let version = dlpk::sys::DLPackVersion::current();

        let mut dl_managed: *mut dlpk::sys::DLManagedTensorVersioned = std::ptr::null_mut();
        unsafe {
            let as_dlpack_fn = array.as_dlpack.expect("labels array must have as_dlpack callback");
            let status = as_dlpack_fn(
                array.ptr as *mut std::os::raw::c_void,
                &mut dl_managed,
                cpu,
                std::ptr::null(),
                version,
            );
            assert!(status == 0, "failed to get DLPack from labels array");
            assert!(!dl_managed.is_null(), "DLPack tensor is null");

            let data_ptr = (*dl_managed).dl_tensor.data as *const LabelValue;
            // The DLPack tensor points into the LabelsValuesArray's data, which
            // lives as long as the Labels. We can free the DLPack metadata
            // immediately since we only need the data pointer.
            if let Some(deleter) = (*dl_managed).deleter {
                deleter(dl_managed);
            }

            std::slice::from_raw_parts(data_ptr, count * size)
        }
    }
}

impl Labels {
    /// Get a pointer to the underlying `mts_labels_t`
    pub(crate) fn as_mts_labels_t(&self) -> *const mts_labels_t {
        self.ptr
    }

    /// Create a new set of `Labels` from a raw `*mut mts_labels_t` pointer.
    ///
    /// This function takes ownership of the pointer and will call
    /// `mts_labels_free` on it when dropped.
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and returned by one of the metatensor-core
    /// functions that create `mts_labels_t`.
    #[inline]
    pub unsafe fn from_raw(ptr: *mut mts_labels_t) -> Labels {
        assert!(!ptr.is_null(), "expected mts_labels_t pointer to not be NULL");
        Labels { ptr }
    }
}

impl std::cmp::PartialEq<Labels> for Labels {
    #[inline]
    fn eq(&self, other: &Labels) -> bool {
        self.names() == other.names() && self.values() == other.values()
    }
}

impl std::ops::Index<usize> for Labels {
    type Output = [LabelValue];

    #[inline]
    fn index(&self, i: usize) -> &[LabelValue] {
        let start = i * self.size();
        let stop = (i + 1) * self.size();
        &self.values()[start..stop]
    }
}

/// iterator over [`Labels`] entries
pub struct LabelsIter<'a> {
    /// start of the labels values
    ptr: *const LabelValue,
    /// Current entry index
    cur: usize,
    /// number of entries
    len: usize,
    /// size of an entry/the labels
    chunk_len: usize,
    phantom: std::marker::PhantomData<&'a LabelValue>,
}

impl<'a> Iterator for LabelsIter<'a> {
    type Item = &'a [LabelValue];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur < self.len {
            unsafe {
                // SAFETY: this should be in-bounds
                let data = self.ptr.add(self.cur * self.chunk_len);
                self.cur += 1;
                // SAFETY: the pointer should be valid for 'a
                Some(std::slice::from_raw_parts(data, self.chunk_len))
            }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.cur;
        return (remaining, Some(remaining));
    }
}

impl ExactSizeIterator for LabelsIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl FusedIterator for LabelsIter<'_> {}

impl<'a> IntoIterator for &'a Labels {
    type IntoIter = LabelsIter<'a>;
    type Item = &'a [LabelValue];

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Parallel iterator over entries in a set of [`Labels`]
#[cfg(feature = "rayon")]
#[derive(Debug, Clone)]
pub struct LabelsParIter<'a> {
    chunks: rayon::slice::ChunksExact<'a, LabelValue>,
}

#[cfg(feature = "rayon")]
impl<'a> rayon::iter::ParallelIterator for LabelsParIter<'a> {
    type Item = &'a [LabelValue];

    #[inline]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item> {
        self.chunks.drive_unindexed(consumer)
    }
}

#[cfg(feature = "rayon")]
impl rayon::iter::IndexedParallelIterator for LabelsParIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.chunks.len()
    }

    #[inline]
    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        self.chunks.drive(consumer)
    }

    #[inline]
    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        self.chunks.with_producer(callback)
    }
}

/// Iterator over entries in a set of [`Labels`] as fixed size arrays
#[derive(Debug, Clone)]
pub struct LabelsFixedSizeIter<'a, const N: usize> {
    values: &'a [LabelValue],
}

impl<'a, const N: usize> Iterator for LabelsFixedSizeIter<'a, N> {
    type Item = &'a [LabelValue; N];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.values.is_empty() {
            return None
        }

        let (value, rest) = self.values.split_at(N);
        self.values = rest;
        return Some(value.try_into().expect("wrong size in FixedSizeIter::next"));
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<const N: usize> ExactSizeIterator for LabelsFixedSizeIter<'_, N> {
    #[inline]
    fn len(&self) -> usize {
        self.values.len() / N
    }
}

/// Builder for [`Labels`]
#[derive(Debug, Clone)]
pub struct LabelsBuilder {
    // cf `Labels` for the documentation of the fields
    names: Vec<String>,
    values: Vec<LabelValue>,
}

impl LabelsBuilder {
    /// Create a new empty `LabelsBuilder` with the given `names`
    #[inline]
    pub fn new(names: Vec<&str>) -> LabelsBuilder {
        let n_unique_names = names.iter().collect::<BTreeSet<_>>().len();
        assert!(n_unique_names == names.len(), "invalid labels: the same name is used multiple times");

        LabelsBuilder {
            names: names.into_iter().map(|s| s.into()).collect(),
            values: Vec::new(),
        }
    }

    /// Reserve space for `additional` other entries in the labels.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional * self.names.len());
    }

    /// Get the number of labels in a single value
    #[inline]
    pub fn size(&self) -> usize {
        self.names.len()
    }

    /// Add a single `entry` to this set of labels.
    ///
    /// This function will panic when attempting to add the same `label` more
    /// than once.
    #[inline]
    pub fn add<T>(&mut self, entry: &[T]) where T: Copy + Into<LabelValue> {
        assert_eq!(
            self.size(), entry.len(),
            "wrong size for added label: got {}, but expected {}",
            entry.len(), self.size()
        );

        // SmallVec allows us to convert everything to `LabelValue` without
        // requiring an extra heap allocation
        let entry = entry.iter().copied().map(Into::into).collect::<SmallVec<[LabelValue; 16]>>();
        self.values.extend(&entry);
    }

    /// Common implementation for `finish` and `finish_unchecked`.
    fn finish_with(
        self,
        creator: unsafe extern "C" fn(
            *const *const std::os::raw::c_char,
            usize,
            crate::c_api::mts_array_t,
        ) -> *mut mts_labels_t
    ) -> Labels {
        let mut raw_names = Vec::new();
        let mut raw_names_ptr = Vec::new();

        for name in &self.names {
            let name = CString::new(&**name).expect("name contains a NULL byte");
            raw_names_ptr.push(name.as_ptr());
            raw_names.push(name);
        }

        let names_count = raw_names_ptr.len();
        let count = if names_count == 0 {
            assert!(self.values.is_empty());
            0
        } else {
            self.values.len() / names_count
        };

        // Wrap raw values in an ndarray-backed mts_array_t
        let shape = ndarray::IxDyn(&[count, names_count]);
        let ndarray_values: ndarray::ArcArray<i32, ndarray::IxDyn> =
            ndarray::Array::from_shape_vec(shape, self.values)
                .expect("shape mismatch when creating labels array")
                .into_shared();
        let array: crate::c_api::mts_array_t = (Box::new(ndarray_values) as Box<dyn crate::data::Array>).into();

        let ptr = unsafe {
            creator(
                raw_names_ptr.as_ptr(),
                names_count,
                array,
            )
        };

        if ptr.is_null() {
            let error = unsafe { crate::c_api::mts_last_error() };
            let message = if error.is_null() {
                "failed to create labels".to_string()
            } else {
                unsafe { std::ffi::CStr::from_ptr(error) }.to_string_lossy().into_owned()
            };
            panic!("{}", message);
        }

        unsafe { Labels::from_raw(ptr) }
    }

    /// Finish building the `Labels`.
    ///
    /// This function checks that all entries in the labels are unique.
    #[inline]
    pub fn finish(self) -> Labels {
        self.finish_with(crate::c_api::mts_labels_create)
    }

    /// Finish building the `Labels`, assuming that all entries are unique.
    ///
    /// This is faster than `finish` as it does not perform a uniqueness check
    /// on the labels entries. It is the caller's responsibility to ensure that
    /// entries are unique.
    ///
    /// # Panics
    ///
    /// If the set of names is not valid (contains duplicates or invalid names).
    #[inline]
    pub fn finish_assume_unique(self) -> Labels {
        self.finish_with(crate::c_api::mts_labels_create_assume_unique)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels() {
        let mut builder = LabelsBuilder::new(vec!["foo", "bar"]);
        builder.add(&[2, 3]);
        builder.add(&[1, 243]);
        builder.add(&[-4, -2413]);

        let labels = builder.finish();
        assert_eq!(labels.names(), &["foo", "bar"]);
        assert_eq!(labels.size(), 2);
        assert_eq!(labels.count(), 3);
        assert!(!labels.is_empty());

        assert_eq!(labels[0], [2, 3]);
        assert_eq!(labels[1], [1, 243]);
        assert_eq!(labels[2], [-4, -2413]);

        let builder = LabelsBuilder::new(vec![]);
        let labels = builder.finish();
        assert_eq!(labels.size(), 0);
        assert_eq!(labels.count(), 0);

        let mut builder = LabelsBuilder::new(vec!["foo", "bar"]);
        builder.add(&[2, 3]);
        builder.add(&[1, 243]);
        let labels = builder.finish_assume_unique();
        assert_eq!(labels.names(), &["foo", "bar"]);
        assert_eq!(labels.size(), 2);
        assert_eq!(labels.count(), 2);
    }

    #[test]
    fn direct_construct() {
        let labels = Labels::new(
            ["foo", "bar"],
            &[
                [2, 3],
                [1, 243],
                [-4, -2413],
            ]
        );

        assert_eq!(labels.names(), &["foo", "bar"]);
        assert_eq!(labels.size(), 2);
        assert_eq!(labels.count(), 3);

        assert_eq!(labels[0], [2, 3]);
        assert_eq!(labels[1], [1, 243]);
        assert_eq!(labels[2], [-4, -2413]);
    }

    #[test]
    fn iter() {
        let mut builder = LabelsBuilder::new(vec!["foo", "bar"]);
        builder.add(&[2, 3]);
        builder.add(&[1, 2]);
        builder.add(&[4, 3]);

        let labels = builder.finish();
        let mut iter = labels.iter();
        assert_eq!(iter.len(), 3);

        assert_eq!(iter.next().unwrap(), &[2, 3]);
        assert_eq!(iter.next().unwrap(), &[1, 2]);
        assert_eq!(iter.next().unwrap(), &[4, 3]);
        assert_eq!(iter.next(), None);
    }

    #[cfg( feature = "rayon")]
    #[test]
    fn par_iter() {
        use rayon::iter::IndexedParallelIterator;

        let mut builder = LabelsBuilder::new(vec!["foo", "bar"]);
        builder.add(&[2, 3]);
        builder.add(&[1, 2]);
        builder.add(&[4, 3]);

        let labels = builder.finish();
        let iter = labels.par_iter();
        assert_eq!(iter.len(), 3);

        let mut values = Vec::new();
        iter.collect_into_vec(&mut values);

        assert_eq!(values, [&[2, 3], &[1, 2], &[4, 3]]);
    }

    #[test]
    fn iter_fixed_size() {
        let mut builder = LabelsBuilder::new(vec!["foo", "bar"]);
        builder.add(&[1, 2]);
        builder.add(&[2, 3]);

        let labels = builder.finish();

        for (i, [a, b]) in labels.iter_fixed_size().enumerate() {
            assert_eq!(*a as usize, 1 + i);
            assert_eq!(*b as usize, 2 + i);
        }
    }

    #[test]
    #[should_panic(expected = "wrong label size in `iter_fixed_size`: the entries contains 2 element but this function was called with size of 3")]
    fn iter_fixed_size_wrong_size() {
        let labels = LabelsBuilder::new(vec!["foo", "bar"]).finish();

        for [_, _, _] in labels.iter_fixed_size() {}
    }

    #[test]
    #[should_panic(expected = "'33 bar' is not a valid label name")]
    fn invalid_label_name() {
        LabelsBuilder::new(vec!["foo", "33 bar"]).finish();
    }

    #[test]
    #[should_panic(expected = "invalid labels: the same name is used multiple times")]
    fn duplicated_label_name() {
        LabelsBuilder::new(vec!["foo", "bar", "foo"]).finish();
    }

    #[test]
    #[should_panic(expected = "can not have the same label entry multiple times: [0, 1] is already present")]
    fn duplicated_label_entry() {
        let mut builder = LabelsBuilder::new(vec!["foo", "bar"]);
        builder.add(&[0, 1]);
        builder.add(&[0, 1]);
        builder.finish();
    }

    #[test]
    fn single_label() {
        let labels = Labels::single();
        assert_eq!(labels.names(), &["_"]);
        assert_eq!(labels.size(), 1);
        assert_eq!(labels.count(), 1);
    }

    #[test]
    fn empty_label() {
        let labels = LabelsBuilder::new(vec!["foo", "bar"]).finish();

        assert!(labels.is_empty());
        assert_eq!(labels.count(), 0);
        assert_eq!(labels.size(), 2);
    }

    #[test]
    fn position() {
        let mut builder = LabelsBuilder::new(vec!["foo", "bar"]);
        builder.add(&[1, 2]);
        builder.add(&[2, 3]);
        let labels = builder.finish();

        assert!(labels.contains(&[1, 2]));
        assert_eq!(labels.position(&[1, 2]), Some(0));

        assert!(labels.contains(&[2, 3]));
        assert_eq!(labels.position(&[2, 3]), Some(1));

        assert!(!labels.contains(&[3, 3]));
        assert_eq!(labels.position(&[3, 3]), None);
    }

    #[test]
    fn indexing() {
        let labels = Labels::new(
            ["foo", "bar"],
            &[
                [2, 3],
                [1, 243],
                [-4, -2413],
            ]
        );

        assert_eq!(labels[1], [1, 243]);
        assert_eq!(labels[2], [-4, -2413]);
    }

    #[test]
    fn debug() {
        let labels = Labels::new(
            ["foo", "bar"],
            &[
                [2, 3],
                [1, 243],
                [-4, -2413],
            ]
        );

        let expected = format!(
            "Labels @ {:p} {{\n    foo, bar\n     2    3   \n     1   243  \n    -4   -2413  \n}}\n",
            labels.ptr
        );
        assert_eq!(format!("{:?}", labels), expected);
    }

    #[test]
    fn union() {
        let first = Labels::new(["aa", "bb"], &[[0, 1], [1, 2]]);
        let second = Labels::new(["aa", "bb"], &[[2, 3], [1, 2], [4, 5]]);

        let mut first_mapping = vec![0; first.count()];
        let mut second_mapping = vec![0; second.count()];
        let union = first.union(&second, Some(&mut first_mapping), Some(&mut second_mapping)).unwrap();

        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values(), [0, 1, 1, 2, 2, 3, 4, 5]);

        assert_eq!(first_mapping, [0, 1]);
        assert_eq!(second_mapping, [2, 1, 3]);
    }

    #[test]
    fn intersection() {
        let first = Labels::new(["aa", "bb"], &[[0, 1], [1, 2]]);
        let second = Labels::new(["aa", "bb"], &[[2, 3], [1, 2], [4, 5]]);

        let mut first_mapping = vec![0_i64; first.count()];
        let mut second_mapping = vec![0_i64; second.count()];
        let union = first.intersection(&second, Some(&mut first_mapping), Some(&mut second_mapping)).unwrap();

        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values(), [1, 2]);

        assert_eq!(first_mapping, [-1, 0]);
        assert_eq!(second_mapping, [-1, 0, -1]);
    }

    #[test]
    fn difference() {
        let first = Labels::new(["aa", "bb"], &[[0, 1], [1, 2]]);
        let second = Labels::new(["aa", "bb"], &[[2, 3], [1, 2], [4, 5]]);

        let mut mapping = vec![0_i64; first.count()];
        let union = first.difference(&second, Some(&mut mapping)).unwrap();

        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values(), [0, 1]);

        assert_eq!(mapping, [0, -1]);
    }

    #[test]
    fn selection() {
        // selection with a subset of names
        let labels = Labels::new(["aa", "bb"], &[[1, 1], [1, 2], [3, 2], [2, 1]]);
        let selection = Labels::new(["aa"], &[[1], [2], [5]]);

        let selected = labels.select(&selection).unwrap();
        assert_eq!(selected, [0, 1, 3]);

        // selection with the same names
        let selection = Labels::new(["aa", "bb"], &[[1, 1], [2, 1], [5, 1], [1, 2]]);
        let selected = labels.select(&selection).unwrap();
        assert_eq!(selected, [0, 3, 1]);

        // empty selection
        let selection = Labels::empty(vec!["aa"]);
        let selected = labels.select(&selection).unwrap();
        assert_eq!(selected, []);

        // invalid selection names
        let selection = Labels::empty(vec!["aaa"]);
        let err = labels.select(&selection).unwrap_err();
        assert_eq!(err.message,
            "invalid parameter: 'aaa' in selection is not part of these Labels"
        );
    }
}
