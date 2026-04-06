#![allow(clippy::default_trait_access)]
use std::ffi::CString;
use std::collections::BTreeSet;

use dlpk::{DLDevice, DLDeviceType};
use hashbrown::HashMap;
use hashbrown::hash_map::RawEntryMut;

use once_cell::sync::OnceCell;
use smallvec::SmallVec;

use crate::Error;
use crate::data::mts_array_t;
use crate::utils::ConstCString;

mod array;
use self::array::{create_array_from_vec, load_values_from_array};

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

/// Check if the given dimension is a valid identifier, to be used in `Labels`.
pub fn is_valid_label_dimension(dimension: &str) -> bool {
    if dimension.is_empty() {
        return false;
    }

    for (i, c) in dimension.chars().enumerate() {
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
/// `(labels.count(), labels.size())`, with a set of dimension names associated
/// with the columns of this array. Each row/entry in this array is unique, and
/// they are often (but not always) sorted in  lexicographic order.
///
/// The main way to construct a new set of labels is to use a `LabelsBuilder`.
pub struct Labels {
    /// Dimensions of the labels, stored as const C strings for easier
    /// integration with the C API
    dimensions: Vec<ConstCString>,
    /// The values of the labels, one row per entry. This is stored in a
    /// `mts_array_t` to allow it to live on GPU if needed.
    values: mts_array_t,
    /// CPU values, lazily materialized from array via DLPack
    cpu_values: OnceCell<Vec<LabelValue>>,
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
        if self.dimensions != other.dimensions {
            return false;
        }

        if self.count() != other.count() {
            return false;
        }

        if self.device() != other.device() {
            return false;
        }

        if self.device().device_type == dlpk::DLDeviceType::kDLExtDev {
            // kDLExtDev is used for the meta device in torch, and there is no
            // data to compare, so we consider all Labels on this device as
            // equal (as long as dimensions and count are the same)
            return true
        }

        return self.values_cpu() == other.values_cpu();
    }
}

impl Eq for Labels {}


impl std::fmt::Debug for Labels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Labels{{")?;
        writeln!(f, "    {}", self.dimensions().join(", "))?;

        let widths = self.dimensions().iter().map(|s| s.len()).collect::<Vec<_>>();
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
    pub fn empty(dimensions: &[&str]) -> Result<Labels, Error> {
        return Labels::from_vec(dimensions, Vec::new());
    }

    /// Create new Labels with the given dimension names and values.
    ///
    /// The values are given as a flatten, row-major array, and we will check
    /// that rows are unique in the array.
    pub fn new(dimensions: &[&str], values: mts_array_t) -> Result<Labels, Error> {
        return Labels::new_impl(dimensions, values, true);
    }

    /// Create new labels with the given dimension names and values.
    ///
    /// This is identical to [`Labels::new`] except that the rows are not
    /// checked for uniqueness, but instead the caller must ensure that rows are
    /// unique.
    ///
    /// note: this function still checks for uniqueness when compiled in debug
    /// mode, to help find issues in calling code.
    pub unsafe fn new_unchecked_uniqueness(dimensions: &[&str], values: mts_array_t) -> Result<Labels, Error> {
        if cfg!(debug_assertions) {
            let device = values.device()?;
            if device.device_type == DLDeviceType::kDLExtDev {
                // this device is used as the "meta" device by torch, and there
                // is no data to check for uniqueness even in debug mode.
                return Labels::new_impl(dimensions, values, false);
            }
            return Labels::new_impl(dimensions, values, true);
        } else {
            return Labels::new_impl(dimensions, values, false);
        }
    }

    /// Helper constructor to make tests more readable
    pub fn from_vec(dimensions: &[&str], values: Vec<LabelValue>) -> Result<Labels, Error> {
        return Labels::from_vec_impl(dimensions, values, true);
    }

    pub unsafe fn from_vec_unchecked_uniqueness(dimensions: &[&str], values: Vec<LabelValue>) -> Result<Labels, Error> {
        if cfg!(debug_assertions) {
            return Labels::from_vec_impl(dimensions, values, true);
        } else {
            return Labels::from_vec_impl(dimensions, values, false);
        }
    }

    fn from_vec_impl(dimensions: &[&str], values: Vec<LabelValue>, check_unique: bool) -> Result<Labels, Error> {
        let dimensions = Labels::validate_dimensions(dimensions)?;

        if dimensions.is_empty() {
            assert!(values.is_empty());
            return Ok(Labels {
                dimensions: Vec::new(),
                values: create_array_from_vec(Vec::new(), 0, 0),
                cpu_values: OnceCell::with_value(Vec::new()),
                sorted: OnceCell::with_value(true),
                positions: Default::default(),
            });
        }

        let size = dimensions.len();
        assert!(dimensions.len() % size == 0);
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

        let cpu_values = values.clone();
        let values = create_array_from_vec(values, count, size);

        Ok(Labels {
            dimensions,
            values,
            cpu_values: OnceCell::with_value(cpu_values),
            sorted: OnceCell::new(),
            positions: OnceCell::new(),
        })
    }

    /// Validate dimensions and return `ConstCString` vec
    fn validate_dimensions(dimensions: &[&str]) -> Result<Vec<ConstCString>, Error> {
        for dimension in dimensions {
            if !is_valid_label_dimension(dimension) {
                return Err(Error::InvalidParameter(format!(
                    "'{}' is not a valid label dimension", dimension
                )));
            }
        }

        let mut unique_dimensions = BTreeSet::new();
        for dimension in dimensions {
            if !unique_dimensions.insert(dimension) {
                return Err(Error::InvalidParameter(format!(
                    "label dimensions must be unique, got '{}' multiple times", dimension
                )));
            }
        }

        Ok(dimensions.iter()
            .map(|&s| ConstCString::new(CString::new(s).expect("invalid C string")))
            .collect::<Vec<_>>())
    }

    /// Actual implementation of both [`Labels::new`] and
    /// [`Labels::new_unchecked_uniqueness`]
    fn new_impl(dimensions: &[&str], values: mts_array_t, check_unique: bool) -> Result<Labels, Error> {
        let dimensions = Labels::validate_dimensions(dimensions)?;

        if dimensions.is_empty() {
            assert!(values.shape()?.iter().product::<usize>() == 0);
            return Ok(Labels {
                dimensions: Vec::new(),
                values: create_array_from_vec(Vec::new(), 0, 0),
                cpu_values: OnceCell::with_value(Vec::new()),
                sorted: OnceCell::with_value(true),
                positions: Default::default(),
            });
        }

        let size = dimensions.len();
        assert!(dimensions.len() % size == 0);

        let mut cpu_values = OnceCell::new();
        if check_unique {
            cpu_values = OnceCell::with_value(load_values_from_array(&values, size)?);
            let mut entries = cpu_values.get_mut().expect("we just initialized these").chunks_exact(size).collect::<Vec<_>>();
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
            dimensions,
            values,
            cpu_values,
            sorted: OnceCell::new(),
            positions: OnceCell::new(),
        })
    }

    /// Get the values on CPU, materializinf them from the `mts_array_t` if
    /// needed.
    pub(crate) fn values_cpu(&self) -> &[LabelValue] {
        self.cpu_values.get_or_init(|| {
            match load_values_from_array(&self.values, self.size()) {
                Ok(values) => values,
                Err(err) => std::panic::panic_any(err),
            }
        })
    }

    /// Get the number of entries/named values in a single label
    pub fn size(&self) -> usize {
        self.dimensions.len()
    }

    /// Get the device on which these labels are stored.
    pub fn device(&self) -> DLDevice {
        return self.values.device().expect("failed to get the device for Labels values");
    }

    /// Get the names of the dimensions in this set of labels
    pub fn dimensions(&self) -> Vec<&str> {
        self.dimensions.iter().map(|s| s.as_str()).collect()
    }

    /// Get the names of the dimensions in this set of labels as
    /// C-compatible (null terminated) strings
    pub fn c_dimensions(&self) -> &[ConstCString] {
        &self.dimensions
    }

    /// Check whether entries in these Labels are sorted in lexicographic order,
    /// and cache the result.
    pub fn is_sorted(&self) -> bool {
        *self.sorted.get_or_init(|| is_sorted::IsSorted::is_sorted(&mut self.values_cpu().chunks_exact(self.size())))
    }

    /// Get the backing `mts_array_t` for the label values.
    pub fn values(&self) -> &mts_array_t {
        &self.values
    }

    /// Get the total number of entries in this set of labels.
    pub fn count(&self) -> usize {
        self.values.shape().expect("could not get the shape of labels")[0]
    }

    /// Check if this set of Labels is empty (contains no entry)
    pub fn is_empty(&self) -> bool {
        self.count() == 0
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

    /// Compute the union of two labels, and optionally the mapping from the
    /// position of entries in the inputs to positions of entries in the output.
    ///
    /// Mapping will be computed only if slices are not empty.
    #[allow(clippy::needless_range_loop)]
    pub fn union(&self, other: &Labels, first_mapping: &mut [i64], second_mapping: &mut [i64]) -> Result<Labels, Error> {
        if self.dimensions != other.dimensions {
            return Err(Error::InvalidParameter(
                "can not take the union of these Labels, they have different dimensions".into()
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

        for (i, labels_entry) in other.to_cpu().iter().enumerate() {
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
        let cpu_values = values.clone();
        let values = create_array_from_vec(values, count, size);

        return Ok(Labels {
            dimensions: self.dimensions.clone(),
            values,
            cpu_values: OnceCell::with_value(cpu_values),
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
        if self.dimensions != other.dimensions {
            return Err(Error::InvalidParameter(
                "can not take the intersection of these Labels, they have different dimensions".into()
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
        for (i, entry) in first.to_cpu().iter().enumerate() {
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
        let cpu_values = values.clone();
        let values = create_array_from_vec(values, count, size);

        return Ok(Labels {
            dimensions: self.dimensions.clone(),
            values,
            cpu_values: OnceCell::with_value(cpu_values),
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
        if self.dimensions != other.dimensions {
            return Err(Error::InvalidParameter(
                "can not take the difference of these Labels, they have different dimensions".into(),
            ));
        }

        if !first_mapping.is_empty() {
            assert!(first_mapping.len() == self.count());
            first_mapping.fill(-1);
        }

        let mut values = Vec::new();
        let mut new_position = 0;

        // Loop through the elements of the first set
        for (i, entry) in self.to_cpu().iter().enumerate() {
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
        let cpu_values = values.clone();
        let values = create_array_from_vec(values, count, size);

        return Ok(Labels {
            dimensions: self.dimensions.clone(),
            values: values,
            cpu_values: OnceCell::with_value(cpu_values),
            sorted,
            positions: OnceCell::new(),
        });
    }

    /// Select entries in these `Labels` that match the `selection`.
    ///
    /// The selection's dimensions must be a subset of the dimensions of these
    /// `Labels`.
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
        if selection.dimensions == self.dimensions {
            for entry in &selection.to_cpu() {
                #[allow(clippy::cast_possible_wrap)]
                if let Some(position) = self.position(entry) {
                    selected[n_selected] = position as i64;
                    n_selected += 1;
                }
            }
        } else {
            let mut dimensions_to_match = Vec::new();
            for dimension in &selection.dimensions {
                let i = match self.dimensions.iter().position(|n| n == dimension) {
                    Some(index) => index,
                    None => {
                        return Err(Error::InvalidParameter(format!(
                            "'{}' in selection is not part of these Labels", dimension.as_str()
                        )))
                    }
                };
                dimensions_to_match.push(i);
            }

            let mut candidate = vec![0; dimensions_to_match.len()];
            for (entry_i, entry) in self.to_cpu().iter().enumerate() {
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

    pub fn to_cpu(&self) -> CpuLabels<'_> {
        CpuLabels { labels: self }
    }
}

/// A wrapper around `Labels` that provides easier access to the values on CPU.
///
/// This provides `Index` and `IntoIterator` implementations to access the
/// entries in the labels. It is a separate struct from `Labels` to make it
/// clear when we are accessing values on CPU.
pub struct CpuLabels<'a> {
    labels: &'a Labels,
}

impl CpuLabels<'_> {
    /// Iterate over the entries in these Labels
    pub fn iter(&self) -> Iter<'_> {
        let values = self.values_cpu();
        debug_assert!(values.len() % self.dimensions.len().max(1) == 0);
        return Iter {
            ptr: values.as_ptr(),
            cur: 0,
            len: self.count(),
            chunk_len: self.size(),
            phantom: std::marker::PhantomData,
        };
    }
}

impl std::ops::Deref for CpuLabels<'_> {
    type Target = Labels;

    fn deref(&self) -> &Labels {
        self.labels
    }
}

impl std::ops::Index<usize> for CpuLabels<'_> {
    type Output = [LabelValue];

    #[inline]
    fn index(&self, i: usize) -> &[LabelValue] {
        let start = i * self.size();
        let stop = (i + 1) * self.size();
        &self.values_cpu()[start..stop]
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

impl<'a, 'b> IntoIterator for &'a CpuLabels<'b> where 'b: 'a {
    type IntoIter = Iter<'a>;
    type Item = &'a [LabelValue];
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn valid_dimensions() {
        let e = Labels::empty(&["not an ident"]).err().unwrap();
        assert_eq!(e.to_string(), "invalid parameter: 'not an ident' is not a valid label dimension");

        let e = Labels::empty(&["not", "there", "not"]).err().unwrap();
        assert_eq!(e.to_string(), "invalid parameter: label dimensions must be unique, got 'not' multiple times");
    }

    #[test]
    fn sorted() {
        let labels = Labels::from_vec(&["aa", "bb"],
            vec![0, 1, /**/ 1, 2]
        ).unwrap();

        assert!(labels.sorted.get().is_none());
        assert!(labels.is_sorted());
        assert!(labels.sorted.get().is_some());

        let labels = Labels::from_vec(&["aa", "bb"],
            vec![0, 1, /**/ 1, 2, /**/ 0, 2]
        ).unwrap();

        assert!(!labels.is_sorted());
    }

    #[test]
    fn union() {
        let first = Labels::from_vec(
            &["aa", "bb"],
            vec![0, 1, /**/ 1, 2]
        ).unwrap();

        let second = Labels::from_vec(
            &["aa", "bb"],
            vec![2, 3, /**/ 1, 2, /**/ 4, 5]
        ).unwrap();

        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; second.count()];

        let union = first.union(&second, first_mapping, second_mapping).unwrap();
        assert_eq!(union.dimensions(), ["aa", "bb"]);
        assert_eq!(union.values_cpu(), &[0, 1, 1, 2, 2, 3, 4, 5]);
        assert_eq!(first_mapping, &[0, 1]);
        assert_eq!(second_mapping, &[2, 1, 3]);

        let first_mapping = &mut vec![0; second.count()];
        let second_mapping = &mut vec![0; first.count()];

        let union = second.union(&first, first_mapping, second_mapping).unwrap();
        assert_eq!(union.dimensions(), ["aa", "bb"]);
        assert_eq!(union.values_cpu(), &[2, 3, 1, 2, 4, 5, 0, 1]);
        assert_eq!(first_mapping, &[0, 1, 2]);
        assert_eq!(second_mapping, &[3, 1]);

        let labels = Labels::empty(&["aa"]).unwrap();
        let err = first.union(&labels, &mut [], &mut []).unwrap_err();
        assert_eq!(
            format!("{}", err),
            "invalid parameter: can not take the union of these Labels, they have different dimensions"
        );

        // Take the union with an empty set of labels
        let empty = Labels::empty(&["aa", "bb"]).unwrap();
        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; empty.count()];

        let union = first.union(&empty, first_mapping, second_mapping).unwrap();
        assert_eq!(union.dimensions(), ["aa", "bb"]);
        assert_eq!(union.values_cpu(), &[0, 1, 1, 2]);
        assert_eq!(first_mapping, &[0, 1]);
        assert_eq!(second_mapping, &Vec::<i64>::new());
    }

    #[test]
    fn intersection() {
        let first = Labels::from_vec(
            &["aa", "bb"],
            vec![0, 1, /**/ 1, 2]
        ).unwrap();

        let second = Labels::from_vec(
            &["aa", "bb"],
            vec![2, 3, /**/ 1, 2, /**/ 4, 5]
        ).unwrap();

        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; second.count()];

        let intersection = first.intersection(&second, first_mapping, second_mapping).unwrap();
        assert_eq!(intersection.dimensions(), ["aa", "bb"]);
        assert_eq!(intersection.values_cpu(), &[1, 2]);
        assert_eq!(first_mapping, &[-1, 0]);
        assert_eq!(second_mapping, &[-1, 0, -1]);

        let first_mapping = &mut vec![0; second.count()];
        let second_mapping = &mut vec![0; first.count()];

        let intersection = second.intersection(&first, first_mapping, second_mapping).unwrap();
        assert_eq!(intersection.dimensions(), ["aa", "bb"]);
        assert_eq!(intersection.values_cpu(), &[1, 2]);
        assert_eq!(first_mapping, &[-1, 0, -1]);
        assert_eq!(second_mapping, &[-1, 0]);

        let labels = Labels::empty(&["aa"]).unwrap();
        let err = first.intersection(&labels, &mut [], &mut []).unwrap_err();
        assert_eq!(
            format!("{}", err),
            "invalid parameter: can not take the intersection of these Labels, they have different dimensions"
        );

        // Take the intersection with an empty set of labels
        let empty = Labels::empty(&["aa", "bb"]).unwrap();
        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; empty.count()];

        let intersection = first.intersection(&empty, first_mapping, second_mapping).unwrap();
        assert_eq!(intersection.dimensions(), ["aa", "bb"]);
        assert_eq!(intersection.count(), 0);
        assert_eq!(first_mapping, &[-1, -1]);
        assert_eq!(second_mapping, &Vec::<i64>::new());
    }

    #[test]
    fn difference() {
        let first = Labels::from_vec(&["aa", "bb"], vec![0, 1, /**/ 1, 2]).unwrap();
        let second = Labels::from_vec(&["aa", "bb"], vec![2, 3, /**/ 1, 2, /**/ 4, 5]).unwrap();

        let first_mapping = &mut vec![0; first.count()];

        let difference = first.difference(&second, first_mapping).unwrap();
        assert_eq!(difference.dimensions(), ["aa", "bb"]);
        assert_eq!(difference.values_cpu(), &[0, 1]);
        assert_eq!(first_mapping, &[0, -1]);

        let first_mapping = &mut vec![0; second.count()];

        let difference = second.difference(&first, first_mapping).unwrap();
        assert_eq!(difference.dimensions(), ["aa", "bb"]);
        assert_eq!(difference.values_cpu(), &[2, 3, /**/ 4, 5]);
        assert_eq!(first_mapping, &[0, -1, 1]);

        let labels = Labels::empty(&["aa"]).unwrap();
        let err = first.difference(&labels, &mut []).unwrap_err();
        assert_eq!(
            format!("{}", err),
            "invalid parameter: can not take the difference of these Labels, they have different dimensions"
        );

        // Take the difference with an empty set of labels
        let empty = Labels::empty(&["aa", "bb"]).unwrap();
        let first_mapping = &mut vec![0; first.count()];

        let difference = first.difference(&empty, first_mapping).unwrap();
        assert_eq!(difference.dimensions(), ["aa", "bb"]);
        assert_eq!(difference.count(), first.count());
        assert_eq!(first_mapping, &[0, 1]);
    }

    #[test]
    fn marker_traits() {
        fn use_send(_: impl Send) {}
        fn use_sync(_: impl Sync) {}

        let labels = Arc::new(Labels::from_vec(&["aa", "bb"], vec![0, 1, 1, 2]).unwrap());

        use_send(labels.clone());
        use_sync(labels);
    }

    #[test]
    fn values_array_always_present() {
        let labels = Labels::from_vec(&["x"], vec![1, 2, 3]).unwrap();

        // Array is always present (primary data)
        let arr = labels.values();
        let shape = arr.shape().unwrap();
        assert_eq!(shape, &[3, 1]);
    }

    #[test]
    fn from_array_lazy_values() {
        // Create a labels array, then build Labels from it
        let original = Labels::from_vec(&["x", "y"], vec![1, 2, 3, 4]).unwrap();
        let array = original.values().try_clone().unwrap();

        let labels = Labels::new(&["x", "y"], array).unwrap();
        assert_eq!(labels.count(), 2);
        assert_eq!(labels.size(), 2);

        let cpu_labels = labels.to_cpu();
        assert_eq!(cpu_labels[0], [1, 2]);
        assert_eq!(cpu_labels[1], [3, 4]);
    }

    #[test]
    fn new_unchecked_uniqueness_valid_no_duplicates() {
        let dimensions = &["x", "y"];
        let values = vec![1, 10, 2, 20, 3, 30];
        let labels = Labels::from_vec(dimensions, values.clone()).unwrap();

        let values = labels.values().try_clone().unwrap();
        let labels_safe = Labels::new(dimensions, values).unwrap();

        let values = labels.values().try_clone().unwrap();
        let labels_unchecked = unsafe {
            Labels::new_unchecked_uniqueness(dimensions, values).unwrap()
        };

        let cpu_labels = labels_safe.to_cpu();
        let cpu_labels_unchecked = labels_unchecked.to_cpu();

        assert_eq!(cpu_labels.count(), 3);
        assert_eq!(cpu_labels_unchecked.count(), 3);
        assert_eq!(&cpu_labels[0], &cpu_labels_unchecked[0]);
        assert_eq!(&cpu_labels[1], &cpu_labels_unchecked[1]);
        assert_eq!(&cpu_labels[2], &cpu_labels_unchecked[2]);
    }
}
