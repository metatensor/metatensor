#![allow(clippy::default_trait_access, clippy::module_name_repetitions)]
use std::ffi::CString;
use std::collections::BTreeSet;

use hashbrown::HashMap;
use hashbrown::hash_map::RawEntryMut;

use once_cell::sync::OnceCell;
use smallvec::SmallVec;

use crate::Error;
use crate::data::mts_array_t;
use crate::utils::ConstCString;

/// A single value inside a label. This is represented as a 32-bit signed
/// integer
pub type LabelValue = i32;

// Labels uses `AHash` instead of the default hasher in std since `AHash` is
// much faster and we don't need the cryptographic strength hash from std.
type LabelsHasher = std::hash::BuildHasherDefault<ahash::AHasher>;

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
    /// Number of entries (rows), cached from array shape or values len
    count: usize,
    /// Always-present backing array (primary data source)
    array: mts_array_t,
    /// CPU values, lazily materialized from array via DLPack
    values: OnceCell<Vec<LabelValue>>,
    /// Whether the entries of the labels (i.e. the rows of the 2D array) are
    /// sorted in lexicographical order. This is lazily initialized on first
    /// access.
    sorted: OnceCell<bool>,
    /// Store the position of all the known labels, for faster access later.
    /// This is lazily initialized whenever a function requires access to the
    /// positions of different entries, allowing to skip the construction of the
    /// `HashMap` when Labels are only used as data storage.
    positions: OnceCell<HashMap<LabelsEntry, usize, LabelsHasher>>,
}

impl PartialEq for Labels {
    fn eq(&self, other: &Self) -> bool {
        self.names == other.names && self.count == other.count && self.values_cpu() == other.values_cpu()
    }
}

impl Eq for Labels {}


impl std::fmt::Debug for Labels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Labels{{")?;
        writeln!(f, "    {}", self.names().join(", "))?;

        let widths = self.names().iter().map(|s| s.len()).collect::<Vec<_>>();
        let values = self.values_cpu();
        for entry in values.chunks_exact(self.size()) {
            write!(f, "    ")?;
            for (value, width) in entry.iter().zip(&widths) {
                write!(f, "{:^width$}  ", value, width=width)?;
            }
            writeln!(f)?;
        }

        writeln!(f, "}}")?;
        Ok(())
    }
}

fn init_positions(values: &[LabelValue], size: usize) -> HashMap<LabelsEntry, usize, LabelsHasher> {
    assert!(values.len() % size == 0);

    let mut positions = HashMap::with_hasher(LabelsHasher::default());
    for (i, entry) in values.chunks_exact(size).enumerate() {
        // entries should be unique!
        unsafe {
            positions.insert_unique_unchecked(entry.into(), i);
        }
    }
    return positions;
}

impl Labels {
    /// Create new Labels with the given names and values.
    ///
    /// The values are given as a flatten, row-major array, and we will check
    /// that rows are unique in the array.
    pub fn new(names: &[&str], values: Vec<LabelValue>) -> Result<Labels, Error> {
        return Labels::new_impl(names, values, true);
    }

    /// Create new labels with the given names and values.
    ///
    /// This is identical to [`Labels::new`] except that the rows are not
    /// checked for uniqueness, but instead the caller must ensure that rows are
    /// unique.
    ///
    /// note: this function still checks for uniqueness when compiled in debug
    /// mode, to help find issues in calling code.
    pub unsafe fn new_unchecked_uniqueness(names: &[&str], values: Vec<LabelValue>) -> Result<Labels, Error> {
        if cfg!(debug_assertions) {
            return Labels::new_impl(names, values, true);
        } else {
            return Labels::new_impl(names, values, false);
        }
    }

    /// Create new Labels from an existing `mts_array_t`.
    ///
    /// If the array is on CPU, values are materialized and uniqueness is
    /// verified. If the array is on a non-CPU device, this returns an error
    /// (use `from_array_assume_unique` for GPU arrays).
    pub fn from_array(names: &[&str], array: mts_array_t) -> Result<Labels, Error> {
        let names_vec = Labels::validate_names(names)?;

        let shape = array.shape()?;
        if shape.len() != 2 {
            return Err(Error::InvalidParameter(
                "labels array must be 2-dimensional".into()
            ));
        }
        if shape[1] != names_vec.len() {
            return Err(Error::InvalidParameter(format!(
                "labels array has {} columns, but {} names were given",
                shape[1], names_vec.len()
            )));
        }

        let count = shape[0];

        if names_vec.is_empty() && count > 0 {
            return Err(Error::InvalidParameter(
                "can not have labels.count > 0 if labels.size is 0".into()
            ));
        }

        // check if data lives on CPU
        let device = array.device()?;
        let cpu = dlpk::sys::DLDevice::cpu();
        if device.device_type != cpu.device_type || device.device_id != cpu.device_id {
            return Err(Error::InvalidParameter(
                "can not verify uniqueness of labels on non-CPU device, \
                use Labels::from_array_assume_unique instead".into()
            ));
        }

        // CPU array: materialize values and check uniqueness
        let values = crate::labels_array::materialize_values_from_array(&array, names_vec.len());

        // check uniqueness
        if !names_vec.is_empty() && count > 0 {
            let size = names_vec.len();
            let mut entries = values.chunks_exact(size).collect::<Vec<_>>();
            entries.sort_unstable();
            if let Some(identical) = entries.windows(2).position(|w| w[0] == w[1]) {
                let entry = entries[identical];
                let entry_display = entry.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                return Err(Error::InvalidParameter(format!(
                    "can not have the same label entry multiple times: [{}] is already present",
                    entry_display
                )));
            }
        }

        Ok(Labels {
            names: names_vec,
            count,
            array,
            values: OnceCell::with_value(values),
            sorted: OnceCell::new(),
            positions: OnceCell::new(),
        })
    }

    /// Create new Labels from an existing `mts_array_t` without checking
    /// uniqueness.
    ///
    /// The caller must ensure that the rows are unique. Passing non-unique
    /// entries is UB (can cause crashes or infinite loops).
    ///
    /// Debug builds still assert uniqueness for CPU arrays.
    pub unsafe fn from_array_assume_unique(names: &[&str], array: mts_array_t) -> Result<Labels, Error> {
        let names_vec = Labels::validate_names(names)?;

        let shape = array.shape()?;
        if shape.len() != 2 {
            return Err(Error::InvalidParameter(
                "labels array must be 2-dimensional".into()
            ));
        }
        if shape[1] != names_vec.len() {
            return Err(Error::InvalidParameter(format!(
                "labels array has {} columns, but {} names were given",
                shape[1], names_vec.len()
            )));
        }

        let count = shape[0];

        if names_vec.is_empty() && count > 0 {
            return Err(Error::InvalidParameter(
                "can not have labels.count > 0 if labels.size is 0".into()
            ));
        }

        // In debug builds, if on CPU, materialize and assert uniqueness
        if cfg!(debug_assertions) {
            let is_cpu = array.device().map_or(false, |device| {
                let cpu = dlpk::sys::DLDevice::cpu();
                device.device_type == cpu.device_type && device.device_id == cpu.device_id
            });
            if is_cpu {
                let values = crate::labels_array::materialize_values_from_array(&array, names_vec.len());
                if !names_vec.is_empty() && count > 0 {
                    let size = names_vec.len();
                    let mut entries = values.chunks_exact(size).collect::<Vec<_>>();
                    entries.sort_unstable();
                    if let Some(identical) = entries.windows(2).position(|w| w[0] == w[1]) {
                        let entry = entries[identical];
                        let entry_display = entry.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                        panic!(
                            "non-unique labels entry in from_array_assume_unique: [{}]",
                            entry_display
                        );
                    }
                }

                return Ok(Labels {
                    names: names_vec,
                    count,
                    array,
                    values: OnceCell::with_value(values),
                    sorted: OnceCell::new(),
                    positions: OnceCell::new(),
                });
            }
        }

        Ok(Labels {
            names: names_vec,
            count,
            array,
            values: OnceCell::new(),
            sorted: OnceCell::new(),
            positions: OnceCell::new(),
        })
    }

    /// Helper constructor to make tests more readable
    #[cfg(test)]
    pub fn new_i32(names: &[&str], values: Vec<i32>) -> Result<Labels, Error> {
        return Labels::new(names, values);
    }

    /// Validate names and return ConstCString vec
    fn validate_names(names: &[&str]) -> Result<Vec<ConstCString>, Error> {
        for name in names {
            if !is_valid_label_name(name) {
                return Err(Error::InvalidParameter(format!(
                    "'{}' is not a valid label name", name
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

        Ok(names.iter()
            .map(|&s| ConstCString::new(CString::new(s).expect("invalid C string")))
            .collect::<Vec<_>>())
    }

    /// Actual implementation of both [`Labels::new`] and
    /// [`Labels::new_unchecked_uniqueness`]
    fn new_impl(names: &[&str], values: Vec<LabelValue>, check_unique: bool) -> Result<Labels, Error> {
        let names = Labels::validate_names(names)?;

        if names.is_empty() {
            assert!(values.is_empty());
            let array = crate::labels_array::LabelsValuesArray::from_vec(
                Vec::new(), 0, 0,
            );
            return Ok(Labels {
                names: Vec::new(),
                count: 0,
                array,
                values: OnceCell::with_value(Vec::new()),
                sorted: OnceCell::with_value(true),
                positions: Default::default(),
            });
        }

        let size = names.len();
        assert!(values.len() % size == 0);
        let count = values.len() / size;

        if check_unique {
            let mut entries = values.chunks_exact(size).collect::<Vec<_>>();
            entries.sort_unstable();
            if let Some(identical) = entries.windows(2).position(|w| w[0] == w[1]) {
                let entry = entries[identical];
                let entry_display = entry.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                return Err(Error::InvalidParameter(format!(
                    "can not have the same label entry multiple times: [{}] is already present",
                    entry_display
                )));
            }
        }

        let array = crate::labels_array::LabelsValuesArray::from_vec(
            values.clone(), count, size,
        );

        Ok(Labels {
            names,
            count,
            array,
            values: OnceCell::with_value(values),
            sorted: OnceCell::new(),
            positions: OnceCell::new(),
        })
    }

    /// Lazily get the CPU values, materializing from the array if needed
    pub(crate) fn values_cpu(&self) -> &[LabelValue] {
        self.values.get_or_init(|| {
            crate::labels_array::materialize_values_from_array(&self.array, self.size())
        })
    }

    /// Pre-fill the cached CPU values without triggering materialization
    /// from the array. Used when the caller has values from a known-good
    /// source (e.g., device transfer where the source Labels was validated).
    ///
    /// If the values are already cached, this is a no-op.
    pub fn set_cached_values(&self, values: Vec<LabelValue>) {
        let _ = self.values.set(values);
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

    /// Check whether entries in these Labels are sorted in lexicographic order,
    /// and cache the result.
    pub fn is_sorted(&self) -> bool {
        *self.sorted.get_or_init(|| is_sorted::IsSorted::is_sorted(&mut self.values_cpu().chunks_exact(self.size())))
    }

    /// Get the backing `mts_array_t` for the label values.
    pub fn values(&self) -> &mts_array_t {
        &self.array
    }

    /// Get the total number of entries in this set of labels
    pub fn count(&self) -> usize {
        self.count
    }

    /// Check if this set of Labels is empty (contains no entry)
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check whether the given `label` is part of this set of labels
    pub fn contains(&self, label: &[LabelValue]) -> bool {
        let positions = self.positions.get_or_init(|| init_positions(self.values_cpu(), self.size()));
        positions.contains_key(label)
    }

    /// Get the position (i.e. row index) of the given label in the full labels
    /// array, or None.
    pub fn position(&self, value: &[LabelValue]) -> Option<usize> {
        assert!(value.len() == self.size(), "invalid size of index in Labels::position");

        return self.get_or_init_positions().get(value).copied();
    }

    fn get_or_init_positions(&self) -> &HashMap<LabelsEntry, usize, LabelsHasher> {
        return self.positions.get_or_init(|| init_positions(self.values_cpu(), self.size()));
    }

    /// Iterate over the entries in these Labels
    pub fn iter(&self) -> Iter<'_> {
        let values = self.values_cpu();
        debug_assert!(values.len() % self.names.len().max(1) == 0);
        return Iter {
            ptr: values.as_ptr(),
            cur: 0,
            len: self.count,
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
        let mut values = self.values_cpu().to_vec();

        if !first_mapping.is_empty() {
            assert!(first_mapping.len() == self.count());
            #[allow(clippy::cast_possible_wrap)]
            for i in 0..self.count() {
                first_mapping[i] = i as i64;
            }
        }

        for (i, labels_entry) in other.iter().enumerate() {
            let labels_entry = labels_entry.iter().copied().collect::<LabelsEntry>();

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

        let size = self.size();
        let count = if size == 0 { 0 } else { values.len() / size };
        let array = crate::labels_array::LabelsValuesArray::from_vec(
            values.clone(), count, size,
        );

        return Ok(Labels {
            names: self.names.clone(),
            count,
            array,
            values: OnceCell::with_value(values),
            sorted: OnceCell::new(),
            positions: OnceCell::with_value(positions),
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

        let sorted = if first.sorted.get() == Some(&true) {
            // if the input was sorted, the output will be as well, since we
            // can only remove entries
            OnceCell::with_value(true)
        } else {
            // we'll need to check, since the removed entries could be the ones
            // out of order
            OnceCell::new()
        };

        let size = self.size();
        let count = if size == 0 { 0 } else { values.len() / size };
        let array = crate::labels_array::LabelsValuesArray::from_vec(
            values.clone(), count, size,
        );

        return Ok(Labels {
            names: self.names.clone(),
            count,
            array,
            values: OnceCell::with_value(values),
            sorted,
            positions: OnceCell::new(),
        });
    }

    /// Compute the difference of two labels, and optionally the mapping from
    /// the position of entries in the inputs to positions of entries in the
    /// output.
    ///
    /// Mapping will be computed only if slices are not empty.
    pub fn difference(&self, other: &Labels, first_mapping: &mut [i64]) -> Result<Labels, Error> {
        if self.names != other.names {
            return Err(Error::InvalidParameter(
                "can not take the difference of these Labels, they have different names".into(),
            ));
        }

        if !first_mapping.is_empty() {
            assert!(first_mapping.len() == self.count());
            first_mapping.fill(-1);
        }

        let mut values = Vec::new();
        let mut new_position = 0;

        // Loop through the elements of the first set
        for (i, entry) in self.iter().enumerate() {
            // Check whether the entry is an element of the second set
            if !other.contains(entry) {
                // If they are not present, append the values to the set difference
                values.extend_from_slice(entry);

                // Fill the first mapping with the position of the element in
                // the set difference.
                if !first_mapping.is_empty() {
                    first_mapping[i] = new_position;
                }

                new_position += 1;
            }
        }

        let sorted = if self.sorted.get() == Some(&true) {
            // if the input was sorted, the output will be as well, since we
            // can only remove entries
            OnceCell::with_value(true)
        } else {
            // we'll need to check, since the removed entries could be the ones
            // out of order
            OnceCell::new()
        };

        let size = self.size();
        let count = if size == 0 { 0 } else { values.len() / size };
        let array = crate::labels_array::LabelsValuesArray::from_vec(
            values.clone(), count, size,
        );

        return Ok(Labels {
            names: self.names.clone(),
            count,
            array,
            values: OnceCell::with_value(values),
            sorted,
            positions: OnceCell::new(),
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

            let mut candidate = vec![0; dimensions_to_match.len()];
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
        &self.values_cpu()[start..stop]
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn valid_names() {
        let e = Labels::new(&["not an ident"], Vec::<LabelValue>::new()).err().unwrap();
        assert_eq!(e.to_string(), "invalid parameter: 'not an ident' is not a valid label name");

        let e = Labels::new(&["not", "there", "not"], Vec::<LabelValue>::new()).err().unwrap();
        assert_eq!(e.to_string(), "invalid parameter: labels names must be unique, got 'not' multiple times");
    }

    #[test]
    fn sorted() {
        let labels = Labels::new_i32(&["aa", "bb"],
            vec![0, 1, /**/ 1, 2]
        ).unwrap();

        assert!(!labels.sorted.get().is_some());
        assert!(labels.is_sorted());
        assert!(labels.sorted.get().is_some());

        let labels = Labels::new_i32(&["aa", "bb"],
            vec![0, 1, /**/ 1, 2, /**/ 0, 2]
        ).unwrap();

        assert!(!labels.is_sorted());
    }

    #[test]
    fn union() {
        let first = Labels::new_i32(
            &["aa", "bb"],
            vec![0, 1, /**/ 1, 2]
        ).unwrap();

        let second = Labels::new_i32(
            &["aa", "bb"],
            vec![2, 3, /**/ 1, 2, /**/ 4, 5]
        ).unwrap();

        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; second.count()];

        let union = first.union(&second, first_mapping, second_mapping).unwrap();
        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values_cpu(), &[0, 1, 1, 2, 2, 3, 4, 5]);
        assert_eq!(first_mapping, &[0, 1]);
        assert_eq!(second_mapping, &[2, 1, 3]);

        let first_mapping = &mut vec![0; second.count()];
        let second_mapping = &mut vec![0; first.count()];

        let union = second.union(&first, first_mapping, second_mapping).unwrap();
        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values_cpu(), &[2, 3, 1, 2, 4, 5, 0, 1]);
        assert_eq!(first_mapping, &[0, 1, 2]);
        assert_eq!(second_mapping, &[3, 1]);

        let labels = Labels::new(&["aa"], Vec::<LabelValue>::new()).unwrap();
        let err = first.union(&labels, &mut [], &mut []).unwrap_err();
        assert_eq!(
            format!("{}", err),
            "invalid parameter: can not take the union of these Labels, they have different names"
        );

        // Take the union with an empty set of labels
        let empty = Labels::new(&["aa", "bb"], Vec::<LabelValue>::new()).unwrap();
        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; empty.count()];

        let union = first.union(&empty, first_mapping, second_mapping).unwrap();
        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values_cpu(), &[0, 1, 1, 2]);
        assert_eq!(first_mapping, &[0, 1]);
        assert_eq!(second_mapping, &Vec::<i64>::new());
    }

    #[test]
    fn intersection() {
        let first = Labels::new_i32(
            &["aa", "bb"],
            vec![0, 1, /**/ 1, 2]
        ).unwrap();

        let second = Labels::new_i32(
            &["aa", "bb"],
            vec![2, 3, /**/ 1, 2, /**/ 4, 5]
        ).unwrap();

        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; second.count()];

        let intersection = first.intersection(&second, first_mapping, second_mapping).unwrap();
        assert_eq!(intersection.names(), ["aa", "bb"]);
        assert_eq!(intersection.values_cpu(), &[1, 2]);
        assert_eq!(first_mapping, &[-1, 0]);
        assert_eq!(second_mapping, &[-1, 0, -1]);

        let first_mapping = &mut vec![0; second.count()];
        let second_mapping = &mut vec![0; first.count()];

        let intersection = second.intersection(&first, first_mapping, second_mapping).unwrap();
        assert_eq!(intersection.names(), ["aa", "bb"]);
        assert_eq!(intersection.values_cpu(), &[1, 2]);
        assert_eq!(first_mapping, &[-1, 0, -1]);
        assert_eq!(second_mapping, &[-1, 0]);

        let labels = Labels::new(&["aa"], Vec::<LabelValue>::new()).unwrap();
        let err = first.intersection(&labels, &mut [], &mut []).unwrap_err();
        assert_eq!(
            format!("{}", err),
            "invalid parameter: can not take the intersection of these Labels, they have different names"
        );

        // Take the intersection with an empty set of labels
        let empty = Labels::new(&["aa", "bb"], Vec::<LabelValue>::new()).unwrap();
        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; empty.count()];

        let intersection = first.intersection(&empty, first_mapping, second_mapping).unwrap();
        assert_eq!(intersection.names(), ["aa", "bb"]);
        assert_eq!(intersection.count(), 0);
        assert_eq!(first_mapping, &[-1, -1]);
        assert_eq!(second_mapping, &Vec::<i64>::new());
    }

    #[test]
    fn difference() {
        let first = Labels::new_i32(&["aa", "bb"], vec![0, 1, /**/ 1, 2]).unwrap();
        let second = Labels::new_i32(&["aa", "bb"], vec![2, 3, /**/ 1, 2, /**/ 4, 5]).unwrap();

        let first_mapping = &mut vec![0; first.count()];

        let difference = first.difference(&second, first_mapping).unwrap();
        assert_eq!(difference.names(), ["aa", "bb"]);
        assert_eq!(difference.values_cpu(), &[0, 1]);
        assert_eq!(first_mapping, &[0, -1]);

        let first_mapping = &mut vec![0; second.count()];

        let difference = second.difference(&first, first_mapping).unwrap();
        assert_eq!(difference.names(), ["aa", "bb"]);
        assert_eq!(difference.values_cpu(), &[2, 3, /**/ 4, 5]);
        assert_eq!(first_mapping, &[0, -1, 1]);

        let labels = Labels::new(&["aa"], Vec::<LabelValue>::new()).unwrap();
        let err = first.difference(&labels, &mut []).unwrap_err();
        assert_eq!(
            format!("{}", err),
            "invalid parameter: can not take the difference of these Labels, they have different names"
        );

        // Take the difference with an empty set of labels
        let empty = Labels::new(&["aa", "bb"], Vec::<LabelValue>::new()).unwrap();
        let first_mapping = &mut vec![0; first.count()];

        let difference = first.difference(&empty, first_mapping).unwrap();
        assert_eq!(difference.names(), ["aa", "bb"]);
        assert_eq!(difference.count(), first.count());
        assert_eq!(first_mapping, &[0, 1]);
    }

    #[test]
    fn marker_traits() {
        fn use_send(_: impl Send) {}
        fn use_sync(_: impl Sync) {}

        let labels = Arc::new(Labels::new_i32(&["aa", "bb"], vec![0, 1, 1, 2]).unwrap());

        use_send(labels.clone());
        use_sync(labels);
    }

    #[test]
    fn values_array_always_present() {
        let labels = Labels::new_i32(&["x"], vec![1, 2, 3]).unwrap();

        // Array is always present (primary data)
        let arr = labels.values();
        let shape = arr.shape().unwrap();
        assert_eq!(shape, &[3, 1]);
    }

    #[test]
    fn from_array_lazy_values() {
        // Create a labels array, then build Labels from it
        let original = Labels::new_i32(&["x", "y"], vec![1, 2, 3, 4]).unwrap();
        let array = original.values().try_clone().unwrap();

        let labels = Labels::from_array(&["x", "y"], array).unwrap();
        assert_eq!(labels.count(), 2);
        assert_eq!(labels.size(), 2);
        assert_eq!(labels[0], [1, 2]);
        assert_eq!(labels[1], [3, 4]);
    }

    #[test]
    fn new_unchecked_uniqueness_valid_no_duplicates() {
        let names = &["x", "y"];
        let values = vec![1, 10, 2, 20, 3, 30];

        let labels_safe = Labels::new(names, values.clone()).unwrap();
        let labels_unchecked = unsafe { Labels::new_unchecked_uniqueness(names, values.clone()).unwrap() };

        assert_eq!(labels_safe.count(), 3);
        assert_eq!(labels_unchecked.count(), 3);
        assert_eq!(&labels_safe[0], &labels_unchecked[0]);
        assert_eq!(&labels_safe[1], &labels_unchecked[1]);
        assert_eq!(&labels_safe[2], &labels_unchecked[2]);
    }
}
