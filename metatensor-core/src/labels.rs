#![allow(clippy::default_trait_access, clippy::module_name_repetitions)]
use std::sync::RwLock;
use std::ffi::CString;
use std::collections::BTreeSet;
use std::os::raw::c_void;

use hashbrown::HashMap;
use hashbrown::hash_map::RawEntryMut;

use once_cell::sync::OnceCell;
use smallvec::SmallVec;

use crate::Error;
use crate::utils::ConstCString;

/// A single value inside a label. This is represented as a 32-bit signed
/// integer, with a couple of helper function to get its value as usize/isize.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct LabelValue(i32);

impl PartialEq<i32> for LabelValue {
    fn eq(&self, other: &i32) -> bool {
        self.0 == *other
    }
}

impl PartialEq<LabelValue> for i32 {
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

#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
impl From<usize> for LabelValue {
    fn from(value: usize) -> LabelValue {
        assert!(value < i32::MAX as usize);
        LabelValue(value as i32)
    }
}

#[allow(clippy::cast_possible_truncation)]
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
        self.0
    }
}

// Labels uses `AHash` instead of the default hasher in std since `AHash` is
// much faster and we don't need the cryptographic strength hash from std.
type AHashHasher = std::hash::BuildHasherDefault<ahash::AHasher>;

// Use a small vec to store Labels entries in the `positions`. This helps by
// removing heap allocations in the most common case (fewer than 8 dimensions in
// the Labels), while still allowing the Labels to contains many dimensions.
type LabelsEntry = SmallVec<[LabelValue; 8]>;

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

#[derive(Debug)]
struct UserData {
    ptr: *mut c_void,
    delete: Option<unsafe extern fn(*mut c_void)>,
}

impl UserData {
    /// Create an empty `UserData`
    fn null() -> UserData {
        UserData {
            ptr: std::ptr::null_mut(),
            delete: None,
        }
    }
}

impl Drop for UserData {
    fn drop(&mut self) {
        if let Some(delete) = self.delete {
            unsafe {
                delete(self.ptr);
            }
        }
    }
}

// These should be enforced by the code using metatensor
unsafe impl Sync for UserData {}
unsafe impl Send for UserData {}

/// A set of labels used to carry metadata associated with a tensor map.
///
/// This is similar to a list of named tuples, but stored as a 2D array of shape
/// `(labels.count(), labels.size())`, with a of set names associated with the
/// columns of this array. Each row/entry in this array is unique, and they are
/// often (but not always) sorted in  lexicographic order.
///
/// The main way to construct a new set of labels is to use a `LabelsBuilder`.
pub struct Labels {
    /// Names of the labels, stored as const C strings for easier integration
    /// with the C API
    names: Vec<ConstCString>,
    /// Values of the labels, as a linearized 2D array in row-major order
    values: Vec<LabelValue>,
    /// Whether the entries of the labels (i.e. the rows of the 2D array) are
    /// sorted in lexicographical order
    sorted: bool,
    /// Store the position of all the known labels, for faster access later.
    /// This is lazily initialized whenever a function requires access to the
    /// positions of different entries, allowing to skip the construction of the
    /// `HashMap` when Labels are only used as data storage.
    positions: OnceCell<HashMap<LabelsEntry, usize, AHashHasher>>,
    /// Some data provided by the user that we should keep around (this is
    /// used to store a pointer to the on-GPU tensor in metatensor-torch).
    user_data: RwLock<UserData>,
}

impl PartialEq for Labels {
    fn eq(&self, other: &Self) -> bool {
        self.names == other.names && self.values == other.values
    }
}

impl Eq for Labels {}


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

fn init_positions(values: &[LabelValue], size: usize) -> HashMap<LabelsEntry, usize, AHashHasher> {
    assert!(values.len() % size == 0);

    let mut positions = HashMap::new();
    for (i, entry) in values.chunks_exact(size).enumerate() {
        // entries should be unique!
        positions.insert_unique_unchecked(entry.into(), i);
    }
    return positions;
}

impl Labels {
    /// Create new Labels with the given names and values.
    ///
    /// The values are given as a flatten, row-major array, and we will check
    /// that rows are unique in the array.
    pub fn new(names: &[&str], values: Vec<impl Into<LabelValue>>) -> Result<Labels, Error> {
        let values = values.into_iter().map(Into::into).collect();
        return Labels::new_impl(names, values, true);
    }

    /// Create new labels with the given names and values.
    ///
    /// This is identical to [`Labels::new`] except that the rows are not
    /// checked for uniqueness, but instead the caller must ensure that rows are
    /// unique.
    pub unsafe fn new_unchecked_uniqueness(names: &[&str], values: Vec<impl Into<LabelValue>>) -> Result<Labels, Error> {
        let values = values.into_iter().map(Into::into).collect();
        if cfg!(debug_assertions) {
            return Labels::new_impl(names, values, true);
        } else {
            return Labels::new_impl(names, values, false);
        }
    }

    /// Actual implementation of both [`Labels::new`] and
    /// [`Labels::new_unchecked_uniqueness`]
    fn new_impl(names: &[&str], values: Vec<LabelValue>, check_unique: bool) -> Result<Labels, Error> {
        for name in names {
            if !is_valid_label_name(name) {
                return Err(Error::InvalidParameter(format!(
                    "all labels names must be valid identifiers, '{}' is not", name
                )));
            }
        }

        let mut unique_names = BTreeSet::new();
        for name in names {
            if !unique_names.insert(name) {
                return Err(Error::InvalidParameter(format!(
                    "labels names must be unique, got '{}' multiple times", name
                )));
            }
        }

        let names = names.iter()
            .map(|&s| ConstCString::new(CString::new(s).expect("invalid C string")))
            .collect::<Vec<_>>();

        if names.is_empty() {
            assert!(values.is_empty());
            return Ok(Labels {
                names: Vec::new(),
                values: Vec::new(),
                sorted: true,
                positions: Default::default(),
                user_data: RwLock::new(UserData::null()),
            });
        }

        let size = names.len();
        assert!(values.len() % size == 0);

        if check_unique {
            let mut entries = values.chunks_exact(size).collect::<Vec<_>>();
            entries.sort_unstable();
            if let Some(identical) = entries.windows(2).position(|w| w[0] == w[1]) {
                let entry = entries[identical];
                let entry_display = entry.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                return Err(Error::InvalidParameter(format!(
                    "can not have the same label entry multiple time: [{}] is already present",
                    entry_display
                )));
            }
        }

        let sorted = is_sorted::IsSorted::is_sorted(&mut values.chunks_exact(size));
        Ok(Labels {
            names: names,
            values: values,
            sorted: sorted,
            positions: OnceCell::new(),
            user_data: RwLock::new(UserData::null()),
        })
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

    /// Check whether entries in these Labels are sorted in lexicographic order
    pub fn is_sorted(&self) -> bool {
        self.sorted
    }

    /// Get the registered user data (this will be NULL if no data was
    /// registered)
    pub fn user_data(&self) -> *mut c_void {
        let guard = self.user_data.read().expect("poisoned lock");
        return guard.ptr;
    }

    /// Register user data for these Labels.
    ///
    /// The `user_data_delete` will be called with `user_data` when the Labels
    /// are dropped, and should free the memory associated with `user_data`.
    ///
    /// Any existing user data will be released (by calling the provided
    /// `user_data_delete` function) before overwriting with the new data.
    pub fn set_user_data(
        &self,
        user_data: *mut c_void,
        user_data_delete: Option<unsafe extern fn(*mut c_void)>,
    ) {
        let mut guard = self.user_data.write().expect("poisoned lock");
        *guard = UserData {
            ptr: user_data,
            delete: user_data_delete,
        };
    }

    /// Get the total number of entries in this set of labels
    pub fn count(&self) -> usize {
        if self.size() == 0 {
            return 0;
        } else {
            return self.values.len() / self.size();
        }
    }

    /// Check if this set of Labels is empty (contains no entry)
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Check whether the given `label` is part of this set of labels
    pub fn contains(&self, label: &[LabelValue]) -> bool {
        let positions = self.positions.get_or_init(|| init_positions(&self.values, self.size()));
        positions.contains_key(label)
    }

    /// Get the position (i.e. row index) of the given label in the full labels
    /// array, or None.
    pub fn position(&self, value: &[LabelValue]) -> Option<usize> {
        assert!(value.len() == self.size(), "invalid size of index in Labels::position");

        return self.get_or_init_positions().get(value).copied();
    }

    fn get_or_init_positions(&self) -> &HashMap<LabelsEntry, usize, AHashHasher> {
        return self.positions.get_or_init(|| init_positions(&self.values, self.size()));
    }

    /// Iterate over the entries in these Labels
    pub fn iter(&self) -> Iter {
        debug_assert!(self.values.len() % self.names.len() == 0);
        return Iter {
            ptr: self.values.as_ptr(),
            cur: 0,
            len: self.count(),
            chunk_len: self.size(),
            phantom: std::marker::PhantomData,
        };
    }

    /// Compute the union of two labels, and optionally the mapping from the
    /// position of entries in the inputs to positions of entries in the output.
    ///
    /// Mapping will be computed only if slices are not empty.
    #[allow(clippy::needless_range_loop)]
    pub fn union(&self, other: &Labels, first_mapping: &mut [i64], second_mapping: &mut [i64]) -> Result<Labels, Error> {
        if self.names != other.names {
            return Err(Error::InvalidParameter(
                "can not take the union of these Labels, they have different names".into()
            ));
        }

        let mut positions = self.get_or_init_positions().clone();
        let mut values = self.values.clone();

        if !first_mapping.is_empty() {
            assert!(first_mapping.len() == self.count());
            #[allow(clippy::cast_possible_wrap)]
            for i in 0..self.count() {
                first_mapping[i] = i as i64;
            }
        }

        for (i, labels_entry) in other.iter().enumerate() {
            let labels_entry = labels_entry.iter().copied().map(Into::into).collect::<LabelsEntry>();

            let new_position = positions.len();
            let index = match positions.raw_entry_mut().from_key(&labels_entry) {
                RawEntryMut::Occupied(entry) => *entry.get(),
                RawEntryMut::Vacant(entry) => {
                    values.extend(&labels_entry);
                    entry.insert(labels_entry, new_position);
                    new_position
                }
            };

            #[allow(clippy::cast_possible_wrap)]
            if !second_mapping.is_empty() {
                second_mapping[i] = index as i64;
            }
        }

        let sorted = is_sorted::IsSorted::is_sorted(&mut values.chunks_exact(self.size()));
        return Ok(Labels {
            names: self.names.clone(),
            values,
            sorted: sorted,
            positions: OnceCell::with_value(positions),
            user_data: RwLock::new(UserData::null()),
        });
    }

    /// Compute the intersection of two labels, and optionally the mapping from
    /// the position of entries in the inputs to positions of entries in the
    /// output.
    ///
    /// Mapping will be computed only if slices are not empty.
    pub fn intersection(&self, other: &Labels, first_mapping: &mut [i64], second_mapping: &mut [i64]) -> Result<Labels, Error> {
        if self.names != other.names {
            return Err(Error::InvalidParameter(
                "can not take the intersection of these Labels, they have different names".into()
            ));
        }

        // make `first` the Labels with fewest entries
        let (first, first_indexes, second, second_indexes) = if self.count() <= other.count() {
            (&self, first_mapping, &other, second_mapping)
        } else {
            (&other, second_mapping, &self, first_mapping)
        };

        if !first_indexes.is_empty() {
            assert!(first_indexes.len() == first.count());
            first_indexes.fill(-1);
        }

        if !second_indexes.is_empty() {
            assert!(second_indexes.len() == second.count());
            second_indexes.fill(-1);
        }

        let mut values = Vec::new();
        let mut new_position = 0;
        for (i, entry) in first.iter().enumerate() {
            if let Some(position) = second.position(entry) {
                values.extend_from_slice(entry);

                if !first_indexes.is_empty() {
                    first_indexes[i] = new_position;
                }

                if !second_indexes.is_empty() {
                    second_indexes[position] = new_position;
                }

                new_position += 1;
            }
        }

        let sorted = if first.is_sorted() {
            // if the input was sorted, the output will be as well, since we
            // can only remove entries
            true
        } else {
            // we need to check, since the removed entries could be the ones out
            // of order
            is_sorted::IsSorted::is_sorted(&mut values.chunks_exact(self.size()))
        };

        return Ok(Labels {
            names: self.names.clone(),
            values,
            sorted: sorted,
            positions: OnceCell::new(),
            user_data: RwLock::new(UserData::null()),
        });
    }

    /// Select entries in these `Labels` that match the `selection`.
    ///
    /// The selection's names must be a subset of the name of these `Labels`
    /// names.
    ///
    /// All entries in `self` that match one of the entry in the selection for
    /// all the selection's dimension will be picked. Any entry in the selection
    /// but not in `self` will be ignored.
    ///
    /// On input, selected should have space for `self.count()` elements. On
    /// output, it will contain the indexes in `self` that match the selection.
    /// This function returns the number of selected entries, i.e. the number of
    /// valid indexes in `selected`.
    pub fn select(&self, selection: &Labels, selected: &mut [i64]) -> Result<usize, Error> {
        assert!(selected.len() == self.count());
        selected.fill(-1);

        let mut n_selected = 0;
        if selection.names == self.names {
            for entry in selection {
                #[allow(clippy::cast_possible_wrap)]
                if let Some(position) = self.position(entry) {
                    selected[n_selected] = position as i64;
                    n_selected += 1;
                }
            }
        } else {
            let mut dimensions_to_match = Vec::new();
            for name in &selection.names {
                let i = match self.names.iter().position(|n| n == name) {
                    Some(index) => index,
                    None => {
                        return Err(Error::InvalidParameter(format!(
                            "'{}' in selection is not part of these Labels", name.as_str()
                        )))
                    }
                };
                dimensions_to_match.push(i);
            }

            let mut candidate = vec![LabelValue::new(0); dimensions_to_match.len()];
            for (entry_i, entry) in self.iter().enumerate() {
                for (i, &d) in dimensions_to_match.iter().enumerate() {
                    candidate[i] = entry[d];
                }

                #[allow(clippy::cast_possible_wrap)]
                if selection.contains(&candidate) {
                    selected[n_selected] = entry_i as i64;
                    n_selected += 1;
                }
            }
        }

        return Ok(n_selected);
    }
}

/// iterator over `Labels` entries
pub struct Iter<'a> {
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

impl<'a> Iterator for Iter<'a> {
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

impl ExactSizeIterator for Iter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.len
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

    #[inline]
    fn index(&self, i: usize) -> &[LabelValue] {
        let start = i * self.size();
        let stop = (i + 1) * self.size();
        &self.values[start..stop]
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn valid_names() {
        let e = Labels::new(&["not an ident"], Vec::<i32>::new()).err().unwrap();
        assert_eq!(e.to_string(), "invalid parameter: all labels names must be valid identifiers, 'not an ident' is not");

        let e = Labels::new(&["not", "there", "not"], Vec::<i32>::new()).err().unwrap();
        assert_eq!(e.to_string(), "invalid parameter: labels names must be unique, got 'not' multiple times");
    }

    #[test]
    fn sorted() {
        let labels = Labels::new(&["aa", "bb"],
            vec![0, 1, /**/ 1, 2]
        ).unwrap();

        assert!(labels.is_sorted());

        let labels = Labels::new(&["aa", "bb"],
            vec![0, 1, /**/ 1, 2, /**/ 0, 2]
        ).unwrap();

        assert!(!labels.is_sorted());
    }

    #[test]
    fn union() {
        let first = Labels::new(
            &["aa", "bb"],
            vec![0, 1, /**/ 1, 2]
        ).unwrap();

        let second = Labels::new(
            &["aa", "bb"],
            vec![2, 3, /**/ 1, 2, /**/ 4, 5]
        ).unwrap();

        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; second.count()];

        let union = first.union(&second, first_mapping, second_mapping).unwrap();
        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values, &[0, 1, 1, 2, 2, 3, 4, 5]);
        assert_eq!(first_mapping, &[0, 1]);
        assert_eq!(second_mapping, &[2, 1, 3]);

        let first_mapping = &mut vec![0; second.count()];
        let second_mapping = &mut vec![0; first.count()];

        let union = second.union(&first, first_mapping, second_mapping).unwrap();
        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values, &[2, 3, 1, 2, 4, 5, 0, 1]);
        assert_eq!(first_mapping, &[0, 1, 2]);
        assert_eq!(second_mapping, &[3, 1]);

        let labels = Labels::new(&["aa"], Vec::<i32>::new()).unwrap();
        let err = first.union(&labels, &mut [], &mut []).unwrap_err();
        assert_eq!(
            format!("{}", err),
            "invalid parameter: can not take the union of these Labels, they have different names"
        );

        // Take the union with an empty set of labels
        let empty = Labels::new(&["aa", "bb"], Vec::<i32>::new()).unwrap();
        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; empty.count()];

        let union = first.union(&empty, first_mapping, second_mapping).unwrap();
        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values, &[0, 1, 1, 2]);
        assert_eq!(first_mapping, &[0, 1]);
        assert_eq!(second_mapping, &[]);
    }

    #[test]
    fn intersection() {
        let first = Labels::new(
            &["aa", "bb"],
            vec![0, 1, /**/ 1, 2]
        ).unwrap();

        let second = Labels::new(
            &["aa", "bb"],
            vec![2, 3, /**/ 1, 2, /**/ 4, 5]
        ).unwrap();

        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; second.count()];

        let intersection = first.intersection(&second, first_mapping, second_mapping).unwrap();
        assert_eq!(intersection.names(), ["aa", "bb"]);
        assert_eq!(intersection.values, &[1, 2]);
        assert_eq!(first_mapping, &[-1, 0]);
        assert_eq!(second_mapping, &[-1, 0, -1]);

        let first_mapping = &mut vec![0; second.count()];
        let second_mapping = &mut vec![0; first.count()];

        let intersection = second.intersection(&first, first_mapping, second_mapping).unwrap();
        assert_eq!(intersection.names(), ["aa", "bb"]);
        assert_eq!(intersection.values, &[1, 2]);
        assert_eq!(first_mapping, &[-1, 0, -1]);
        assert_eq!(second_mapping, &[-1, 0]);

        let labels = Labels::new(&["aa"], Vec::<i32>::new()).unwrap();
        let err = first.intersection(&labels, &mut [], &mut []).unwrap_err();
        assert_eq!(
            format!("{}", err),
            "invalid parameter: can not take the intersection of these Labels, they have different names"
        );

        // Take the intersection with an empty set of labels
        let empty = Labels::new(&["aa", "bb"], Vec::<i32>::new()).unwrap();
        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; empty.count()];

        let intersection = first.intersection(&empty, first_mapping, second_mapping).unwrap();
        assert_eq!(intersection.names(), ["aa", "bb"]);
        assert_eq!(intersection.count(), 0);
        assert_eq!(first_mapping, &[-1, -1]);
        assert_eq!(second_mapping, &[]);
    }

    #[test]
    fn marker_traits() {
        // ensure Arc<Labels> is Send and Sync, assuming the user data is
        fn use_send(_: impl Send) {}
        fn use_sync(_: impl Sync) {}

        let labels = Arc::new(Labels::new(&["aa", "bb"], vec![0, 1, 1, 2]).unwrap());

        use_send(labels.clone());
        use_sync(labels);
    }
}
