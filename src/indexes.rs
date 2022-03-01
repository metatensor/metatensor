#![allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
#![allow(clippy::default_trait_access, clippy::module_name_repetitions)]

use std::ffi::{CString, CStr};
use std::collections::{BTreeSet, HashMap};
use std::collections::hash_map::Entry;
use std::hash::BuildHasherDefault;

use twox_hash::XxHash64;

#[repr(transparent)]
pub struct ConstCString(*const std::os::raw::c_char);

impl ConstCString {
    pub fn new(str: CString) -> ConstCString {
        ConstCString(CString::into_raw(str))
    }

    pub fn as_c_str(&self) -> &CStr {
        // SAFETY: `CStr::from_ptr` is OK since we created this pointer with
        // `CString::into_raw`, which fulfils all the requirements of
        // `CStr::from_ptr`
        unsafe {
            CStr::from_ptr(self.0)
        }
    }

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


#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IndexValue(i32);

impl std::fmt::Debug for IndexValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Display for IndexValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u32> for IndexValue {
    fn from(value: u32) -> IndexValue {
        assert!(value < i32::MAX as u32);
        IndexValue(value as i32)
    }
}

impl From<i32> for IndexValue {
    fn from(value: i32) -> IndexValue {
        IndexValue(value)
    }
}

impl From<usize> for IndexValue {
    fn from(value: usize) -> IndexValue {
        assert!(value < i32::MAX as usize);
        IndexValue(value as i32)
    }
}

impl From<isize> for IndexValue {
    fn from(value: isize) -> IndexValue {
        assert!(value < i32::MAX as isize && value > i32::MIN as isize);
        IndexValue(value as i32)
    }
}

impl IndexValue {
    #[allow(clippy::cast_sign_loss)]
    pub fn usize(self) -> usize {
        debug_assert!(self.0 >= 0);
        self.0 as usize
    }

    pub fn isize(self) -> isize {
        self.0 as isize
    }

    pub fn i32(self) -> i32 {
        self.0 as i32
    }
}

pub struct IndexesBuilder {
    /// Names of the indexes
    names: Vec<String>,
    /// Values of the indexes, as a linearized 2D array in row-major order
    values: Vec<IndexValue>,
    positions: HashMap<Vec<IndexValue>, usize, BuildHasherDefault<XxHash64>>,
}

impl IndexesBuilder {
    /// Create a new empty `IndexesBuilder` with the given `names`
    pub fn new(names: Vec<&str>) -> IndexesBuilder {
        for name in &names {
            assert!(is_valid_index_name(name), "all indexes names must be valid identifiers, '{}' is not", name);
        }

        let n_unique_names = names.iter().collect::<BTreeSet<_>>().len();
        assert!(n_unique_names == names.len(), "invalid indexes: the same name is used multiple times");

        IndexesBuilder {
            names: names.into_iter().map(|s| s.into()).collect(),
            values: Vec::new(),
            positions: Default::default(),
        }
    }

    /// Get the number of indexes in a single value
    pub fn size(&self) -> usize {
        self.names.len()
    }

    /// Add a single entry with the given `values` for this set of indexes
    pub fn add(&mut self, values: Vec<IndexValue>) {
        assert_eq!(
            self.size(), values.len(),
            "wrong size for added index: got {}, but expected {}", values.len(), self.size()
        );

        self.values.extend(&values);

        let new_position = self.positions.len();
        match self.positions.entry(values) {
            Entry::Occupied(entry) => {
                let values_display = entry.key().iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                panic!(
                    "can not have the same index value multiple time: [{}] is already present at position {}",
                    values_display, entry.get()
                );
            },
            Entry::Vacant(entry) => {
                entry.insert(new_position);
            }
        }
    }

    pub fn contains(&self, values: &[IndexValue]) -> bool {
        self.positions.contains_key(values)
    }

    pub fn finish(self) -> Indexes {
        if self.names.is_empty() {
            assert!(self.values.is_empty());
            return Indexes {
                names: Vec::new(),
                values: Vec::new(),
                positions: Default::default(),
            }
        }

        let names = self.names.into_iter()
            .map(|s| ConstCString::new(CString::new(s).expect("invalid C string")))
            .collect::<Vec<_>>();

        return Indexes {
            names: names,
            values: self.values,
            positions: self.positions,
        };
    }
}

/// Check if the given name is a valid index variable name, to be used as a
/// column name in `Indexes`.
pub fn is_valid_index_name(name: &str) -> bool {
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

#[derive(Clone, PartialEq)]
pub struct Indexes {
    /// Names of the indexes, stored as const C strings for easier integration
    /// with the C API
    names: Vec<ConstCString>,
    /// Values of the indexes, as a linearized 2D array in row-major order
    values: Vec<IndexValue>,
    /// Store the position of all the known indexes, for faster access later.
    /// This uses `XxHash64` instead of the default hasher in std since
    /// `XxHash64` is much faster and we don't need the cryptographic strength
    /// hash from std.
    positions: HashMap<Vec<IndexValue>, usize, BuildHasherDefault<XxHash64>>,
}

impl std::fmt::Debug for Indexes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Indexes{{")?;
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

impl Indexes {
    pub fn single() -> Indexes {
        let mut builder = IndexesBuilder::new(vec!["single_entry"]);
        builder.add(vec![IndexValue::from(0)]);

        return builder.finish();
    }

    /// Get the number of indexes in a single value
    pub fn size(&self) -> usize {
        self.names.len()
    }

    /// Names of the indexes
    pub fn names(&self) -> Vec<&str> {
        self.names.iter().map(|s| s.as_str()).collect()
    }

    /// Names of the indexes as C-compatible (null terminated) strings
    pub fn c_names(&self) -> &[ConstCString] {
        &self.names
    }

    /// How many entries of indexes do we have
    pub fn count(&self) -> usize {
        return self.values.len() / self.size();
    }

    pub fn iter(&self) -> Iter {
        debug_assert!(self.values.len() % self.names.len() == 0);
        return Iter {
            size: self.names.len(),
            values: &self.values
        };
    }

    /// Check whether the given `value` is part of this set of indexes
    pub fn contains(&self, value: &[IndexValue]) -> bool {
        self.position(value).is_some()
    }

    /// Get the position of the given value on this set of indexes, or None.
    pub fn position(&self, value: &[IndexValue]) -> Option<usize> {
        assert!(value.len() == self.size(), "invalid size of index in Indexes::position");

        self.positions.get(value).copied()
    }
}

pub struct Iter<'a> {
    size: usize,
    values: &'a [IndexValue],
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a[IndexValue];
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

impl<'a> IntoIterator for &'a Indexes {
    type IntoIter = Iter<'a>;
    type Item = &'a [IndexValue];
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl std::ops::Index<usize> for Indexes {
    type Output = [IndexValue];
    fn index(&self, i: usize) -> &[IndexValue] {
        let start = i * self.size();
        let stop = (i + 1) * self.size();
        &self.values[start..stop]
    }
}
