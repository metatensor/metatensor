#![allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
#![allow(clippy::default_trait_access, clippy::module_name_repetitions)]

use std::ffi::{CString, CStr};
use std::collections::{BTreeSet, HashMap};
use std::collections::hash_map::Entry;
use std::hash::BuildHasherDefault;

use twox_hash::XxHash64;

/// An analog to `std::ffi::CString` that is immutable & can be shared between
/// traits safely. This is used to store the columns names in a set of `Labels`
/// in a C-compatible way.
#[repr(transparent)]
pub struct ConstCString(*const std::os::raw::c_char);

impl ConstCString {
    /// Create a new `ConstCString` containing the same data as the given `str`.
    pub fn new(str: CString) -> ConstCString {
        ConstCString(CString::into_raw(str))
    }

    /// Get the content of this `ConstCString` as a `Cstr` reference
    pub fn as_c_str(&self) -> &CStr {
        // SAFETY: `CStr::from_ptr` is OK since we created this pointer with
        // `CString::into_raw`, which fulfils all the requirements of
        // `CStr::from_ptr`
        unsafe {
            CStr::from_ptr(self.0)
        }
    }

    /// Get the content of this `ConstCString` as a `str` reference, panicking
    /// if this `ConstCString` contains invalid UTF8.
    pub fn as_str(&self) -> &str {
        return self.as_c_str().to_str().expect("invalid UTF8");
    }
}

impl Drop for ConstCString {
    fn drop(&mut self) {
        // SAFETY: `CString::from_raw` is OK since we created this pointer with
        // `CString::into_raw`
        unsafe {
            let str = CString::from_raw(self.0 as *mut _);
            drop(str);
        }
    }
}

impl PartialEq for ConstCString {
    fn eq(&self, other: &Self) -> bool {
        self.as_c_str() == other.as_c_str()
    }
}

impl Clone for ConstCString {
    fn clone(&self) -> Self {
        let str = self.as_c_str().to_owned();
        return ConstCString::new(str);
    }
}

// SAFETY: Sync is ok since `ConstCString` is immutable, so sharing it between
// threads causes no issue
unsafe impl Sync for ConstCString {}

/// A single value inside a label. This is represented as a 32-bit signed
/// integer, with a couple of helper function to get its value as usize/isize.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct LabelValue(i32);

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

impl From<u32> for LabelValue {
    fn from(value: u32) -> LabelValue {
        assert!(value < i32::MAX as u32);
        LabelValue(value as i32)
    }
}

impl From<i32> for LabelValue {
    fn from(value: i32) -> LabelValue {
        LabelValue(value)
    }
}

impl From<usize> for LabelValue {
    fn from(value: usize) -> LabelValue {
        assert!(value < i32::MAX as usize);
        LabelValue(value as i32)
    }
}

impl From<isize> for LabelValue {
    fn from(value: isize) -> LabelValue {
        assert!(value < i32::MAX as isize && value > i32::MIN as isize);
        LabelValue(value as i32)
    }
}

impl LabelValue {
    /// Create a `LabelValue` with the given `value`
    pub fn new(value: i32) -> LabelValue {
        LabelValue(value)
    }

    /// Get the integer value of this `LabelValue` as a usize
    #[allow(clippy::cast_sign_loss)]
    pub fn usize(self) -> usize {
        debug_assert!(self.0 >= 0);
        self.0 as usize
    }

    /// Get the integer value of this `LabelValue` as an isize
    pub fn isize(self) -> isize {
        self.0 as isize
    }

    /// Get the integer value of this `LabelValue` as an i32
    pub fn i32(self) -> i32 {
        self.0 as i32
    }
}

/// Builder for `Labels`
pub struct LabelsBuilder {
    // cf `Labels` for the documentation of the fields
    names: Vec<String>,
    values: Vec<LabelValue>,
    positions: HashMap<Vec<LabelValue>, usize, BuildHasherDefault<XxHash64>>,
}

impl LabelsBuilder {
    /// Create a new empty `LabelsBuilder` with the given `names`
    pub fn new(names: Vec<&str>) -> LabelsBuilder {
        for name in &names {
            assert!(is_valid_label_name(name), "all labels names must be valid identifiers, '{}' is not", name);
        }

        let n_unique_names = names.iter().collect::<BTreeSet<_>>().len();
        assert!(n_unique_names == names.len(), "invalid labels: the same name is used multiple times");

        LabelsBuilder {
            names: names.into_iter().map(|s| s.into()).collect(),
            values: Vec::new(),
            positions: Default::default(),
        }
    }

    /// Get the number of labels in a single value
    pub fn size(&self) -> usize {
        self.names.len()
    }

    /// Add a single `label` to this set of labels.
    ///
    /// This function will panic when attempting to ad the same `label` more
    /// than once.
    pub fn add(&mut self, label: Vec<LabelValue>) {
        assert_eq!(
            self.size(), label.len(),
            "wrong size for added label: got {}, but expected {}",
            label.len(), self.size()
        );

        self.values.extend(&label);

        let new_position = self.positions.len();
        match self.positions.entry(label) {
            Entry::Occupied(entry) => {
                let values_display = entry.key().iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                panic!(
                    "can not have the same label value multiple time: [{}] is already present at position {}",
                    values_display, entry.get()
                );
            },
            Entry::Vacant(entry) => {
                entry.insert(new_position);
            }
        }
    }

    /// Check if this `LabelBuilder` already contains the given `label`
    pub fn contains(&self, label: &[LabelValue]) -> bool {
        self.positions.contains_key(label)
    }

    /// Finish building the `Labels`
    pub fn finish(self) -> Labels {
        if self.names.is_empty() {
            assert!(self.values.is_empty());
            return Labels {
                names: Vec::new(),
                values: Vec::new(),
                positions: Default::default(),
            }
        }

        let names = self.names.into_iter()
            .map(|s| ConstCString::new(CString::new(s).expect("invalid C string")))
            .collect::<Vec<_>>();

        return Labels {
            names: names,
            values: self.values,
            positions: self.positions,
        };
    }
}

/// Check if the given name is a valid identifier, to be used as a
/// column name in `Labels`.
pub fn is_valid_label_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    for (i, c) in name.chars().enumerate() {
        if i == 0 && c.is_ascii_digit() {
            return false;
        }

        if !(c.is_ascii_alphanumeric() || c == '_') {
            return false;
        }
    }

    return true;
}

/// TODO
#[derive(Clone, PartialEq)]
pub struct Labels {
    /// Names of the labels, stored as const C strings for easier integration
    /// with the C API
    names: Vec<ConstCString>,
    /// Values of the labels, as a linearized 2D array in row-major order
    values: Vec<LabelValue>,
    /// Store the position of all the known labels, for faster access later.
    /// This uses `XxHash64` instead of the default hasher in std since
    /// `XxHash64` is much faster and we don't need the cryptographic strength
    /// hash from std.
    positions: HashMap<Vec<LabelValue>, usize, BuildHasherDefault<XxHash64>>,
}

impl std::fmt::Debug for Labels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Labels{{")?;
        writeln!(f, "    {}", self.names().join(", "))?;

        let widths = self.names().iter().map(|s| s.len()).collect::<Vec<_>>();
        for values in self {
            write!(f, "    ")?;
            for (value, width) in values.iter().zip(&widths) {
                write!(f, "{:^width$}  ", value.isize(), width=width)?;
            }
            writeln!(f)?;
        }

        writeln!(f, "}}")?;
        Ok(())
    }
}

impl Labels {
    /// Create a set of `Labels` containing a single entry, to be used when
    /// there is no relevant information to store.
    pub fn single() -> Labels {
        let mut builder = LabelsBuilder::new(vec!["_"]);
        builder.add(vec![LabelValue::new(0)]);

        return builder.finish();
    }

    /// Get the number of entries/named values in a single label
    pub fn size(&self) -> usize {
        self.names.len()
    }

    /// Get the names of the entries/columns in this set of labels
    pub fn names(&self) -> Vec<&str> {
        self.names.iter().map(|s| s.as_str()).collect()
    }

    /// Get the names of the entries/columns in this set of labels as
    /// C-compatible (null terminated) strings
    pub fn c_names(&self) -> &[ConstCString] {
        &self.names
    }

    /// Get the total number of entries in this set of labels
    pub fn count(&self) -> usize {
        return self.values.len() / self.size();
    }

    /// Iterate over the rows in this set of labels
    pub fn iter(&self) -> Iter {
        debug_assert!(self.values.len() % self.names.len() == 0);
        return Iter {
            size: self.names.len(),
            values: &self.values
        };
    }

    /// Check whether the given `label` is part of this set of labels
    pub fn contains(&self, label: &[LabelValue]) -> bool {
        self.positions.contains_key(label)
    }

    /// Get the position (i.e. row index) of the given label in the full labels
    /// array, or None.
    pub fn position(&self, value: &[LabelValue]) -> Option<usize> {
        assert!(value.len() == self.size(), "invalid size of index in Labels::position");

        self.positions.get(value).copied()
    }
}

/// An iterator over `Labels`
pub struct Iter<'a> {
    size: usize,
    values: &'a [LabelValue],
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a[LabelValue];
    fn next(&mut self) -> Option<Self::Item> {
        if self.values.is_empty() {
            return None
        }

        let (value, rest) = self.values.split_at(self.size);
        self.values = rest;
        return Some(value);
    }
}

impl<'a> ExactSizeIterator for Iter<'a> {
    fn len(&self) -> usize {
        self.values.len() / self.size
    }
}

impl<'a> IntoIterator for &'a Labels {
    type IntoIter = Iter<'a>;
    type Item = &'a [LabelValue];
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl std::ops::Index<usize> for Labels {
    type Output = [LabelValue];
    fn index(&self, i: usize) -> &[LabelValue] {
        let start = i * self.size();
        let stop = (i + 1) * self.size();
        &self.values[start..stop]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes() {
        let mut builder = LabelsBuilder::new(vec!["foo", "bar"]);
        builder.add(vec![LabelValue::new(2), LabelValue::new(3)]);
        builder.add(vec![LabelValue::new(1), LabelValue::new(243)]);
        builder.add(vec![LabelValue::new(-4), LabelValue::new(-2413)]);

        let idx = builder.finish();
        assert_eq!(idx.names(), &["foo", "bar"]);
        assert_eq!(idx.size(), 2);
        assert_eq!(idx.count(), 3);

        assert_eq!(idx[0], [LabelValue::new(2), LabelValue::new(3)]);
        assert_eq!(idx[1], [LabelValue::new(1), LabelValue::new(243)]);
        assert_eq!(idx[2], [LabelValue::new(-4), LabelValue::new(-2413)]);
    }

    #[test]
    fn indexes_iter() {
        let mut builder = LabelsBuilder::new(vec!["foo", "bar"]);
        builder.add(vec![LabelValue::new(2), LabelValue::new(3)]);
        builder.add(vec![LabelValue::new(1), LabelValue::new(2)]);
        builder.add(vec![LabelValue::new(4), LabelValue::new(3)]);

        let idx = builder.finish();
        let mut iter = idx.iter();
        assert_eq!(iter.len(), 3);

        assert_eq!(iter.next().unwrap(), &[LabelValue::new(2), LabelValue::new(3)]);
        assert_eq!(iter.next().unwrap(), &[LabelValue::new(1), LabelValue::new(2)]);
        assert_eq!(iter.next().unwrap(), &[LabelValue::new(4), LabelValue::new(3)]);
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[should_panic(expected = "all labels names must be valid identifiers, \'33 bar\' is not")]
    fn invalid_index_name() {
        LabelsBuilder::new(vec!["foo", "33 bar"]);
    }

    #[test]
    #[should_panic(expected = "invalid labels: the same name is used multiple times")]
    fn duplicated_index_name() {
        LabelsBuilder::new(vec!["foo", "bar", "foo"]);
    }

    #[test]
    #[should_panic(expected = "can not have the same label value multiple time: [0, 1] is already present at position 0")]
    fn duplicated_index_value() {
        let mut builder = LabelsBuilder::new(vec!["foo", "bar"]);
        builder.add(vec![LabelValue::new(0), LabelValue::new(1)]);
        builder.add(vec![LabelValue::new(0), LabelValue::new(1)]);
        builder.finish();
    }

    #[test]
    fn single_label() {
        let labels = Labels::single();
        assert_eq!(labels.names(), &["_"]);
        assert_eq!(labels.size(), 1);
        assert_eq!(labels.count(), 1);
    }
}
