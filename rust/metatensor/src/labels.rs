use std::sync::OnceLock;
use std::ffi::{CStr, CString};
use std::collections::BTreeSet;
use std::iter::FusedIterator;

use crate::MtsArray;
use crate::c_api::mts_labels_t;
use crate::errors::check_ptr;
use crate::errors::{Error, check_status};

/// A single value inside a label.
///
/// This is represented as a 32-bit signed integer, with a couple of helper
/// function to get its value as usize/isize.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct LabelValue(i32);

impl PartialEq<i32> for LabelValue {
    #[inline]
    fn eq(&self, other: &i32) -> bool {
        self.0 == *other
    }
}

impl PartialEq<LabelValue> for i32 {
    #[inline]
    fn eq(&self, other: &LabelValue) -> bool {
        *self == other.0
    }
}

impl std::fmt::Debug for LabelValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Display for LabelValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
impl From<u32> for LabelValue {
    #[inline]
    fn from(value: u32) -> LabelValue {
        assert!(value < i32::MAX as u32);
        LabelValue(value as i32)
    }
}

impl From<i32> for LabelValue {
    #[inline]
    fn from(value: i32) -> LabelValue {
        LabelValue(value)
    }
}

#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
impl From<usize> for LabelValue {
    #[inline]
    fn from(value: usize) -> LabelValue {
        assert!(value < i32::MAX as usize);
        LabelValue(value as i32)
    }
}

#[allow(clippy::cast_possible_truncation)]
impl From<isize> for LabelValue {
    #[inline]
    fn from(value: isize) -> LabelValue {
        assert!(value < i32::MAX as isize && value > i32::MIN as isize);
        LabelValue(value as i32)
    }
}

impl LabelValue {
    /// Create a `LabelValue` with the given `value`
    #[inline]
    pub fn new(value: i32) -> LabelValue {
        LabelValue(value)
    }

    /// Get the integer value of this `LabelValue` as a usize
    #[inline]
    #[allow(clippy::cast_sign_loss)]
    pub fn usize(self) -> usize {
        debug_assert!(self.0 >= 0);
        self.0 as usize
    }

    /// Get the integer value of this `LabelValue` as an isize
    #[inline]
    pub fn isize(self) -> isize {
        self.0 as isize
    }

    /// Get the integer value of this `LabelValue` as an i32
    #[inline]
    pub fn i32(self) -> i32 {
        self.0
    }
}

/// A set of labels used to carry metadata associated with a tensor map.
///
/// This is similar to a list of named tuples, but stored as a 2D array of shape
/// `(labels.count(), labels.size())`, with a of set names associated with the
/// columns of this array. Each row/entry in this array is unique, and they are
/// often (but not always) sorted in  lexicographic order.
///
/// The main way to construct a new set of labels is to use [`Labels::new`] or
/// [`Labels::new_assume_unique`].
///
/// Labels are internally reference counted and immutable, so cloning a `Labels`
/// should be a cheap operation.
pub struct Labels {
    pub(crate) ptr: *const mts_labels_t,
    values_cpu_ptr: OnceLock<*const LabelValue>,
    count: OnceLock<usize>,
    size: OnceLock<usize>,
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
            write!(f, "{:^width$}  ", value.isize(), width=width)?;
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
        Labels {
            ptr,
            values_cpu_ptr: self.values_cpu_ptr.clone(),
            count: self.count.clone(),
            size: self.size.clone(),
        }
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
    /// The `values` can be any type that can be converted into an
    /// [`MtsArray`], including `Vec<[i32; N]>`, `&[[i32; N]]`, or
    /// `ndarray::Array`.
    ///
    /// # Panics
    ///
    /// If the set of names is not valid, or any of the value is duplicated.
    #[inline]
    pub fn new<'a>(names: impl AsRef<[&'a str]>, values: impl Into<MtsArray>) -> Labels {
        Self::new_impl(names.as_ref(), values, crate::c_api::mts_labels)
    }

    /// Create a new set of Labels with the given names and values, without
    /// checking that the entries are unique.
    ///
    /// This is faster than [`Labels::new`] as it does not perform a uniqueness
    /// check on the labels entries. It is the caller's responsibility to ensure
    /// that entries are unique.
    ///
    /// # Panics
    ///
    /// If the set of names is not valid.
    #[inline]
    pub fn new_assume_unique<'a>(names: impl AsRef<[&'a str]>, values: impl Into<MtsArray>) -> Labels {
        Self::new_impl(names.as_ref(), values, crate::c_api::mts_labels_assume_unique)
    }

    /// Common implementation for `new` and `new_assume_unique`
    fn new_impl(
        names: &[&str],
        values: impl Into<MtsArray>,
        creator: unsafe extern "C" fn(
            *const *const std::os::raw::c_char,
            usize,
            crate::c_api::mts_array_t,
        ) -> *const mts_labels_t
    ) -> Labels {
        let n_unique_names = names.iter().collect::<BTreeSet<_>>().len();
        assert!(n_unique_names == names.len(), "invalid labels: the same name is used multiple times");

        let mut raw_names = Vec::new();
        let mut raw_names_ptr = Vec::new();
        for name in names {
            let c_name = CString::new(*name).expect("name contains a NULL byte");
            raw_names_ptr.push(c_name.as_ptr());
            raw_names.push(c_name);
        }

        let array: MtsArray = values.into();
        let ptr = unsafe {
            creator(
                raw_names_ptr.as_ptr(),
                raw_names.len(),
                array.into_raw(),
            )
        };
        check_ptr(ptr).expect("invalid labels");

        unsafe { Labels::from_raw(ptr) }
    }

    /// Get a pointer to the underlying `mts_labels_t`
    pub fn as_mts_labels_t(&self) -> *const mts_labels_t {
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
    pub unsafe fn from_raw(ptr: *const mts_labels_t) -> Labels {
        assert!(!ptr.is_null(), "expected mts_labels_t pointer to not be NULL");
        Labels {
            ptr,
            values_cpu_ptr: OnceLock::new(),
            size: OnceLock::new(),
            count: OnceLock::new(),
        }
    }

    /// Consume the Labels and get the underlying raw pointer.
    ///
    /// After calling this function, the user is responsible for free-ing the
    /// data in `mts_labels_t`, either by re-creating labels with
    /// [`Labels::from_raw`] or passing it to a C API function that will call
    /// [`crate::c_api::mts_labels_free`] on it.
    #[inline]
    pub fn into_raw(mut labels: Labels) -> *const mts_labels_t {
        return std::mem::replace(&mut labels.ptr, std::ptr::null());
    }

    /// Create a set of `Labels` with the given names, containing no entries.
    #[inline]
    pub fn empty<'a>(names: impl AsRef<[&'a str]>) -> Labels {
        let names = names.as_ref();
        let array = ndarray::Array::<i32, _>::from_shape_vec(
            vec![0, names.len()], vec![]
        ).expect("shape mismatch when creating empty labels array");
        Labels::new(names, array)
    }

    /// Create a set of `Labels` containing a single entry, to be used when
    /// there is no relevant information to store.
    #[inline]
    pub fn single() -> Labels {
        Labels::new(["_"], vec![[0i32]])
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
        *self.size.get_or_init(|| self.names().len())
    }

    /// Get the names of the entries/columns in this set of labels
    #[inline]
    pub fn names(&self) -> Vec<&str> {
        let mut names_ptr = std::ptr::null();
        let mut size = 0;
        unsafe {
            check_status(crate::c_api::mts_labels_dimensions(self.ptr, &mut names_ptr, &mut size))
                .expect("failed to get labels dimensions");
        }

        if size == 0 {
            return Vec::new();
        }

        unsafe {
            let names = std::slice::from_raw_parts(names_ptr, size);
            return names.iter()
                        .map(|&ptr| CStr::from_ptr(ptr).to_str().expect("invalid UTF8"))
                        .collect();
        }
    }

    /// Get the values of these labels as a `MtsArray`
    pub fn values(&self) -> MtsArray {
        let mut array = crate::c_api::mts_array_t::null();
        unsafe {
            check_status(crate::c_api::mts_labels_values(
                self.ptr, &mut array,
            )).expect("failed to get labels values array");
        }

        return MtsArray::from_raw(array);
    }

    /// Get the values of these Labels on CPU, potentially copying them from
    /// another device.
    pub fn values_cpu(&self) -> &[LabelValue] {
        let values_cpu_ptr = self.values_cpu_ptr.get_or_init(|| {
            let mut values_cpu_ptr = std::ptr::null();
            let mut count = 0;
            let mut size = 0;

            unsafe {
                check_status(crate::c_api::mts_labels_values_cpu(
                    self.ptr,
                    &mut values_cpu_ptr,
                    &mut count,
                    &mut size
                )).expect("failed to get CPU values for Labels");
            }

            debug_assert_eq!(count, self.count());
            debug_assert_eq!(size, self.size());

            values_cpu_ptr.cast()
        });

        unsafe {
            let count = self.count();
            let size = self.size();
            std::slice::from_raw_parts(*values_cpu_ptr, count * size)
        }
    }

    /// Get the total number of entries in this set of labels
    #[inline]
    pub fn device(&self) -> dlpk::DLDevice {
        let array = self.values();
        return array.device().expect("failed to get the array device");
    }

    /// Get the total number of entries in this set of labels
    #[inline]
    pub fn count(&self) -> usize {
        return *self.count.get_or_init(|| self.values().shape().expect("failed to get the array shape")[0]);
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
        let mut output: *const mts_labels_t = std::ptr::null();
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
        let mut output: *const mts_labels_t = std::ptr::null();
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
        let mut output: *const mts_labels_t = std::ptr::null();
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

    /// Select entries in these `Labels` that match the `selection`.
    ///
    /// The selection's names must be a subset of the names of these labels.
    ///
    /// All entries in these `Labels` that match one of the entry in the
    /// `selection` for all the selection's dimension will be picked. Any entry
    /// in the `selection` but not in these `Labels` will be ignored.
    #[allow(clippy::cast_possible_truncation)]
    pub fn select(&self, selection: &Labels) -> Result<Vec<usize>, Error> {
        let mut selected = vec![0; self.count()];
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

        return Ok(selected.into_iter().map(|s| s as usize).collect());
    }

    /// Iterate over the entries in this set of labels
    #[inline]
    pub fn iter(&self) -> LabelsIter<'_> {
        return LabelsIter {
            ptr: self.values_cpu().as_ptr(),
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
            chunks: self.values_cpu().par_chunks_exact(self.size())
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
            values: self.values_cpu()
        };
    }
}

impl std::cmp::PartialEq<Labels> for Labels {
    #[inline]
    fn eq(&self, other: &Labels) -> bool {
        if self.names() != other.names() {
            return false;
        }

        if self.count() != other.count() {
            return false;
        }

        if self.device() != other.device() {
            return false;
        }

        if self.device().device_type == dlpk::DLDeviceType::kDLExtDev {
            // kDLExtDev is used for torch's meta device, which has no data
            // associated, so we consider all Labels as equal as long as they
            // have the same dimensions and number of entries
            return true;
        } else {
            return self.values_cpu() == other.values_cpu();
        }
    }
}

impl std::ops::Index<usize> for Labels {
    type Output = [LabelValue];

    #[inline]
    fn index(&self, i: usize) -> &[LabelValue] {
        let start = i * self.size();
        let stop = (i + 1) * self.size();
        &self.values_cpu()[start..stop]
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels() {
        let labels = Labels::new(["foo", "bar"], vec![[2, 3], [1, 243], [-4, -2413]]);
        assert_eq!(labels.names(), &["foo", "bar"]);
        assert_eq!(labels.size(), 2);
        assert_eq!(labels.count(), 3);
        assert!(!labels.is_empty());

        assert_eq!(labels[0], [2, 3]);
        assert_eq!(labels[1], [1, 243]);
        assert_eq!(labels[2], [-4, -2413]);

        let labels = Labels::new(&[] as &[&str], Vec::<[i32; 0]>::new());
        assert_eq!(labels.size(), 0);
        assert_eq!(labels.count(), 0);

        let labels = Labels::new_assume_unique(["foo", "bar"], vec![[2, 3], [1, 243]]);
        assert_eq!(labels.names(), &["foo", "bar"]);
        assert_eq!(labels.size(), 2);
        assert_eq!(labels.count(), 2);
    }

    #[test]
    fn direct_construct() {
        let labels = Labels::new(
            ["foo", "bar"],
            vec![[2, 3], [1, 243], [-4, -2413]],
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
        let labels = Labels::new(["foo", "bar"], vec![[2, 3], [1, 2], [4, 3]]);
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

        let labels = Labels::new(["foo", "bar"], vec![[2, 3], [1, 2], [4, 3]]);
        let iter = labels.par_iter();
        assert_eq!(iter.len(), 3);

        let mut values = Vec::new();
        iter.collect_into_vec(&mut values);

        assert_eq!(values, [&[2, 3], &[1, 2], &[4, 3]]);
    }

    #[test]
    fn iter_fixed_size() {
        let labels = Labels::new(["foo", "bar"], vec![[1, 2], [2, 3]]);

        for (i, [a, b]) in labels.iter_fixed_size().enumerate() {
            assert_eq!(a.usize(), 1 + i);
            assert_eq!(b.usize(), 2 + i);
        }
    }

    #[test]
    #[should_panic(expected = "wrong label size in `iter_fixed_size`: the entries contains 2 element but this function was called with size of 3")]
    fn iter_fixed_size_wrong_size() {
        let labels = Labels::new(["foo", "bar"], Vec::<[i32; 2]>::new());

        for [_, _, _] in labels.iter_fixed_size() {}
    }

    #[test]
    #[should_panic(expected = "invalid parameter: '33 bar' is not a valid label name")]
    fn invalid_label_name() {
        Labels::new(["foo", "33 bar"], Vec::<[i32; 2]>::new());
    }

    #[test]
    #[should_panic(expected = "invalid labels: the same name is used multiple times")]
    fn duplicated_label_name() {
        Labels::new(["foo", "bar", "foo"], Vec::<[i32; 3]>::new());
    }

    #[test]
    #[should_panic(expected = "can not have the same label entry multiple times: [0, 1] is already present")]
    fn duplicated_label_entry() {
        Labels::new(["foo", "bar"], vec![[0, 1], [0, 1]]);
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
        let labels = Labels::empty(vec!["foo", "bar"]);

        assert!(labels.is_empty());
        assert_eq!(labels.count(), 0);
        assert_eq!(labels.size(), 2);
    }

    #[test]
    fn position() {
        let labels = Labels::new(["foo", "bar"], vec![[1, 2], [2, 3]]);

        assert!(labels.contains(&[LabelValue::new(1), LabelValue::new(2)]));
        assert_eq!(labels.position(&[LabelValue::new(1), LabelValue::new(2)]), Some(0));

        assert!(labels.contains(&[LabelValue::new(2), LabelValue::new(3)]));
        assert_eq!(labels.position(&[LabelValue::new(2), LabelValue::new(3)]), Some(1));

        assert!(!labels.contains(&[LabelValue::new(3), LabelValue::new(3)]));
        assert_eq!(labels.position(&[LabelValue::new(3), LabelValue::new(3)]), None);
    }

    #[test]
    fn indexing() {
        let labels = Labels::new(
            ["foo", "bar"],
            vec![[2, 3], [1, 243], [-4, -2413]],
        );

        assert_eq!(labels[1], [1, 243]);
        assert_eq!(labels[2], [-4, -2413]);
    }

    #[test]
    fn debug() {
        let labels = Labels::new(
            ["foo", "bar"],
            vec![[2, 3], [1, 243], [-4, -2413]],
        );

        let expected = format!(
            "Labels @ {:p} {{\n    foo, bar\n     2    3   \n     1   243  \n    -4   -2413  \n}}\n",
            labels.ptr
        );
        assert_eq!(format!("{:?}", labels), expected);
    }

    #[test]
    fn union() {
        let first = Labels::new(["aa", "bb"], [[0, 1], [1, 2]]);
        let second = Labels::new(["aa", "bb"], [[2, 3], [1, 2], [4, 5]]);

        let mut first_mapping = vec![0; first.count()];
        let mut second_mapping = vec![0; second.count()];
        let union = first.union(&second, Some(&mut first_mapping), Some(&mut second_mapping)).unwrap();

        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values_cpu(), [0, 1, 1, 2, 2, 3, 4, 5]);

        assert_eq!(first_mapping, [0, 1]);
        assert_eq!(second_mapping, [2, 1, 3]);
    }

    #[test]
    fn intersection() {
        let first = Labels::new(["aa", "bb"], [[0, 1], [1, 2]]);
        let second = Labels::new(["aa", "bb"], [[2, 3], [1, 2], [4, 5]]);

        let mut first_mapping = vec![0_i64; first.count()];
        let mut second_mapping = vec![0_i64; second.count()];
        let union = first.intersection(&second, Some(&mut first_mapping), Some(&mut second_mapping)).unwrap();

        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values_cpu(), [1, 2]);

        assert_eq!(first_mapping, [-1, 0]);
        assert_eq!(second_mapping, [-1, 0, -1]);
    }

    #[test]
    fn difference() {
        let first = Labels::new(["aa", "bb"], [[0, 1], [1, 2]]);
        let second = Labels::new(["aa", "bb"], [[2, 3], [1, 2], [4, 5]]);

        let mut mapping = vec![0_i64; first.count()];
        let union = first.difference(&second, Some(&mut mapping)).unwrap();

        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values_cpu(), [0, 1]);

        assert_eq!(mapping, [0, -1]);
    }

    #[test]
    fn selection() {
        // selection with a subset of names
        let labels = Labels::new(["aa", "bb"], [[1, 1], [1, 2], [3, 2], [2, 1]]);
        let selection = Labels::new(["aa"], [[1], [2], [5]]);

        let selected = labels.select(&selection).unwrap();
        assert_eq!(selected, [0, 1, 3]);

        // selection with the same names
        let selection = Labels::new(["aa", "bb"], [[1, 1], [2, 1], [5, 1], [1, 2]]);
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

    #[test]
    fn labels_into_raw() {
        let original = Labels::new(["foo", "bar"], [[1, 2], [3, 4], [5, 6]]);
        let raw = Labels::into_raw(original);

        let recovered = unsafe { Labels::from_raw(raw) };
        assert_eq!(recovered, Labels::new(["foo", "bar"], [[1, 2], [3, 4], [5, 6]]));
    }
}
