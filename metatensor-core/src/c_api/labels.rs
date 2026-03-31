use std::os::raw::c_char;
use std::ffi::CStr;
use std::sync::Arc;

use crate::{LabelValue, Labels, Error};
use crate::data::mts_array_t;
use super::status::{mts_status_t, catch_unwind};

/// Opaque type representing a set of labels used to carry metadata associated
/// with a tensor map.
///
/// This is similar to a list of `count` named tuples, but stored as a 2D array
/// of shape `(count, size)`, with a set of names associated with the columns of
/// this array (often called *dimensions*). Each row/entry in this array is
/// unique, and they are often (but not always) sorted in lexicographic order.
#[allow(non_camel_case_types)]
pub struct mts_labels_t(Arc<Labels>);

impl mts_labels_t {
    /// Create a raw pointer to `mts_labels_t` using a rust Box wrapping an Arc
    pub fn into_raw(labels: Arc<Labels>) -> *mut mts_labels_t {
        Box::into_raw(Box::new(mts_labels_t(labels)))
    }

    /// Take a raw pointer created by `into_raw` and extract the Arc<Labels>.
    /// The pointer is consumed by this function and no longer valid.
    pub unsafe fn from_raw(ptr: *mut mts_labels_t) -> Arc<Labels> {
        Box::from_raw(ptr).0
    }

    /// Clone the inner Arc without consuming the pointer
    pub fn arc_clone(&self) -> Arc<Labels> {
        Arc::clone(&self.0)
    }
}

impl std::ops::Deref for mts_labels_t {
    type Target = Labels;
    fn deref(&self) -> &Labels {
        &self.0
    }
}


/// Create a new set of Labels from the given dimension names and values
/// array.
///
/// The array must be on a CPU device. This function verifies uniqueness of
/// the labels entries.
///
/// This function allocates memory which must be released with
/// `mts_labels_free` when you don't need it anymore.
///
/// @param names array of NULL-terminated UTF-8 strings containing the names
///        of each dimension
/// @param names_count number of entries in the `names` array
/// @param array the values array (2D, i32, row-major). The labels take
///        ownership of this array.
///
/// @returns A pointer to the newly allocated labels, or a `NULL` pointer in
///          case of error. In case of error, you can use `mts_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern "C" fn mts_labels_create(
    names: *const *const c_char,
    names_count: usize,
    array: mts_array_t,
) -> *mut mts_labels_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);

    let status = catch_unwind(move || {
        let names = create_labels_names_from_raw(names, names_count)?;
        let labels = Labels::from_array(&names, array)?;

        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = mts_labels_t::into_raw(Arc::new(labels));
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}


/// Create a new set of Labels from the given dimension names and values
/// array, without checking for uniqueness of the entries.
///
/// The array can be on any device (CPU or GPU). The caller must ensure
/// that the labels entries are unique; passing non-unique entries is
/// invalid and can lead to crashes or infinite loops.
///
/// This function allocates memory which must be released with
/// `mts_labels_free` when you don't need it anymore.
///
/// @param names array of NULL-terminated UTF-8 strings containing the names
///        of each dimension
/// @param names_count number of entries in the `names` array
/// @param array the values array (2D, i32, row-major). The labels take
///        ownership of this array.
///
/// @returns A pointer to the newly allocated labels, or a `NULL` pointer in
///          case of error. In case of error, you can use `mts_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern "C" fn mts_labels_create_assume_unique(
    names: *const *const c_char,
    names_count: usize,
    array: mts_array_t,
) -> *mut mts_labels_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);

    let status = catch_unwind(move || {
        let names = create_labels_names_from_raw(names, names_count)?;
        let labels = Labels::from_array_assume_unique(&names, array)?;

        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = mts_labels_t::into_raw(Arc::new(labels));
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}


/// Get the dimension names for the given set of `labels`.
///
/// @param labels pointer to an existing set of labels
/// @param names on output, will be set to a pointer to an array of
///        NULL-terminated UTF-8 strings containing the names of each
///        dimension. The array contains `*count` elements.
/// @param count on output, will be set to the number of dimensions
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern "C" fn mts_labels_dimensions(
    labels: *const mts_labels_t,
    names: *mut *const *const c_char,
    count: *mut usize,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(labels, names, count);
        let labels = &*labels;

        *count = labels.size();
        if labels.size() == 0 {
            *names = std::ptr::null();
        } else {
            *names = labels.c_names().as_ptr().cast();
        }

        Ok(())
    })
}



/// Get the values for the given set of `labels` as an `mts_array_t`.
///
/// The returned array is a 2D i32 array with shape `(count, size)`, where
/// `count` is the number of entries and `size` is the number of dimensions.
/// The caller can extract `count` and `size` from the array's shape.
///
/// @param labels pointer to an existing set of labels
/// @param array on output, will be set to the values array. This is a
///        non-owning view (destroy is NULL) valid as long as the labels
///        are alive.
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern "C" fn mts_labels_values(
    labels: *const mts_labels_t,
    array: *mut mts_array_t,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(labels, array);
        let labels = &*labels;

        let values_array = labels.values();
        *array = values_array.raw_copy();

        Ok(())
    })
}


// Internal: raw i32 pointer access for Rust wrapper and C++ labels.
// Not in the public C header (excluded from cbindgen via build.rs).
#[no_mangle]
pub unsafe extern "C" fn mts_labels_values_raw(
    labels: *const mts_labels_t,
    values: *mut *const i32,
    count: *mut usize,
    size: *mut usize,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(labels, values, count, size);
        let labels = &*labels;

        *count = labels.count();
        *size = labels.size();
        if labels.count() == 0 || labels.size() == 0 {
            *values = std::ptr::null();
        } else {
            *values = (&(*labels)[0][0] as *const LabelValue).cast();
        }

        Ok(())
    })
}


/// Get the position of the entry defined by the `values` array in the given set
/// of `labels`.
///
/// @param labels pointer to an existing set of labels
/// @param values array containing the label to lookup
/// @param values_count size of the values array
/// @param result position of the values in the labels or -1 if the values
///               were not found
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
#[allow(clippy::cast_possible_wrap)]
pub unsafe extern "C" fn mts_labels_position(
    labels: *const mts_labels_t,
    values: *const i32,
    values_count: usize,
    result: *mut i64
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(labels, values, result);

        let labels = &*labels;
        if values_count != labels.size() {
            return Err(Error::InvalidParameter(format!(
                "expected label of size {} in mts_labels_position, got size {}",
                labels.size(), values_count
            )));
        }

        assert!(values_count != 0);
        let label = std::slice::from_raw_parts(values.cast(), values_count);
        *result = labels.position(label).map_or(-1, |p| p as i64);

        Ok(())
    })
}



/// Internal helper to create Labels from names array and mts_array_t
unsafe fn create_labels_names_from_raw(
    names_ptr: *const *const c_char,
    names_count: usize,
) -> Result<Vec<&'static str>, Error> {
    if names_count == 0 {
        return Ok(Vec::new());
    }

    if names_ptr.is_null() {
        return Err(Error::InvalidParameter(
            "names can not be NULL".into()
        ));
    }

    let mut names = Vec::with_capacity(names_count);
    for i in 0..names_count {
        let name = CStr::from_ptr(*(names_ptr.add(i)));
        let name = name.to_str().expect("invalid UTF-8 in label name");
        if !crate::labels::is_valid_label_name(name) {
            return Err(Error::InvalidParameter(format!(
                "'{}' is not a valid label name", name
            )));
        }
        names.push(name);
    }

    Ok(names)
}


// Keep old names as aliases for backward compatibility during transition.
// These are excluded from cbindgen via build.rs.
#[no_mangle]
pub unsafe extern "C" fn mts_labels_create_from_array(
    names: *const *const c_char,
    names_count: usize,
    array: mts_array_t,
) -> *mut mts_labels_t {
    mts_labels_create(names, names_count, array)
}

#[no_mangle]
pub unsafe extern "C" fn mts_labels_create_from_array_assume_unique(
    names: *const *const c_char,
    names_count: usize,
    array: mts_array_t,
) -> *mut mts_labels_t {
    mts_labels_create_assume_unique(names, names_count, array)
}


// Internal: used by metatensor-torch for Meta device transfers.
// Not part of the public C API (not in metatensor.h).
#[no_mangle]
pub unsafe extern "C" fn mts_labels_set_cached_values(
    labels: *const mts_labels_t,
    values: *const i32,
    count: usize,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(labels);
        let labels = &*labels;
        let size = labels.size();

        if count != labels.count() {
            return Err(Error::InvalidParameter(format!(
                "mts_labels_set_cached_values: expected count={}, got {}",
                labels.count(), count
            )));
        }

        let n_elements = count * size;
        let label_values = if n_elements > 0 {
            check_pointers_non_null!(values);
            let slice = std::slice::from_raw_parts(values.cast::<LabelValue>(), n_elements);
            slice.to_vec()
        } else {
            Vec::new()
        };

        labels.set_cached_values(label_values);

        Ok(())
    })
}

/// Make a copy of `labels`, incrementing the internal reference count.
///
/// Since `mts_labels_t` are immutable, the copy is actually just a reference
/// count increase, and as such should not be an expensive operation.
///
/// `mts_labels_free` must be used with the returned pointer to decrease the
/// reference count and release the memory when you don't need it anymore.
///
/// @param labels pointer to an existing set of labels
///
/// @returns A pointer to the newly allocated (cloned) labels, or a `NULL`
///          pointer in case of error. In case of error, you can use
///          `mts_last_error()` to get the error message.
#[no_mangle]
pub unsafe extern "C" fn mts_labels_clone(
    labels: *const mts_labels_t,
) -> *mut mts_labels_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);

    let status = catch_unwind(move || {
        check_pointers_non_null!(labels);

        let arc = (*labels).arc_clone();

        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = mts_labels_t::into_raw(arc);
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}


/// common checks and transformations for the set operations
unsafe fn labels_set_common<'a>(
    _operation: &str,
    first: *const mts_labels_t,
    second: *const mts_labels_t,
    first_mapping: *mut i64,
    first_mapping_count: usize,
    second_mapping: *mut i64,
    second_mapping_count: usize,
) -> Result<(&'a mut [i64], &'a mut [i64]), Error> {
    let first_count = (*first).count();
    let second_count = (*second).count();

    let first_mapping = if first_mapping.is_null() {
        &mut []
    } else {
        if first_mapping_count != first_count {
            return Err(Error::InvalidParameter(format!(
                "`first_mapping_count` ({}) must match the number of elements \
                in `first` ({}) but doesn't",
                first_mapping_count,
                first_count,
            )));
        }
        std::slice::from_raw_parts_mut(first_mapping, first_mapping_count)
    };

    let second_mapping = if second_mapping.is_null() {
        &mut []
    } else {
        if second_mapping_count != second_count {
            return Err(Error::InvalidParameter(format!(
                "`second_mapping_count` ({}) must match the number of elements \
                in `second` ({}) but doesn't",
                second_mapping_count,
                second_count,
            )));
        }
        std::slice::from_raw_parts_mut(second_mapping, second_mapping_count)
    };

    return Ok((first_mapping, second_mapping));
}

/// Take the union of two sets of labels.
///
/// If requested, this function can also give the positions in the union where
/// each entry of the input labels ended up.
///
/// This function allocates memory for `*result` which must be released with
/// `mts_labels_free` when you don't need it anymore.
///
/// @param first pointer to the first set of labels
/// @param second pointer to the second set of labels
/// @param result on output, will be set to a pointer to the newly allocated
///        labels containing the union
/// @param first_mapping if you want the mapping from the positions of entries
///        in `first` to the positions in `result`, this should be a pointer
///        to an array containing `first.count` elements, to be filled by this
///        function. Otherwise it should be a `NULL` pointer.
/// @param first_mapping_count number of elements in the `first_mapping` array
/// @param second_mapping if you want the mapping from the positions of entries
///        in `second` to the positions in `result`, this should be a pointer
///        to an array containing `second.count` elements, to be filled by this
///        function. Otherwise it should be a `NULL` pointer.
/// @param second_mapping_count number of elements in the `second_mapping` array
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern "C" fn mts_labels_union(
    first: *const mts_labels_t,
    second: *const mts_labels_t,
    result: *mut *mut mts_labels_t,
    first_mapping: *mut i64,
    first_mapping_count: usize,
    second_mapping: *mut i64,
    second_mapping_count: usize,
) -> mts_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(result);
    catch_unwind(|| {
        check_pointers_non_null!(first, second);

        let (first_mapping, second_mapping) = labels_set_common(
            "union",
            first,
            second,
            first_mapping,
            first_mapping_count,
            second_mapping,
            second_mapping_count
        )?;

        let first_labels: &Labels = &*first;
        let second_labels: &Labels = &*second;

        let result_rust = first_labels.union(
            second_labels,
            first_mapping,
            second_mapping,
        )?;

        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = mts_labels_t::into_raw(Arc::new(result_rust));

        Ok(())
    })
}

/// Take the intersection of two sets of labels.
///
/// If requested, this function can also give the positions in the intersection
/// where each entry of the input labels ended up.
///
/// This function allocates memory for `*result` which must be released with
/// `mts_labels_free` when you don't need it anymore.
///
/// @param first pointer to the first set of labels
/// @param second pointer to the second set of labels
/// @param result on output, will be set to a pointer to the newly allocated
///        labels containing the intersection
/// @param first_mapping if you want the mapping from the positions of entries
///        in `first` to the positions in `result`, this should be a pointer to
///        an array containing `first.count` elements, to be filled by this
///        function. Otherwise it should be a `NULL` pointer. If an entry in
///        `first` is not used in `result`, the mapping will be set to -1.
/// @param first_mapping_count number of elements in the `first_mapping` array
/// @param second_mapping if you want the mapping from the positions of entries
///        in `second` to the positions in `result`, this should be a pointer
///        to an array containing `second.count` elements, to be filled by this
///        function. Otherwise it should be a `NULL` pointer. If an entry in
///        `second` is not used in `result`, the mapping will be set to -1.
/// @param second_mapping_count number of elements in the `second_mapping` array
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern "C" fn mts_labels_intersection(
    first: *const mts_labels_t,
    second: *const mts_labels_t,
    result: *mut *mut mts_labels_t,
    first_mapping: *mut i64,
    first_mapping_count: usize,
    second_mapping: *mut i64,
    second_mapping_count: usize,
) -> mts_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(result);
    catch_unwind(|| {
        check_pointers_non_null!(first, second);

        let (first_mapping, second_mapping) = labels_set_common(
            "intersection",
            first,
            second,
            first_mapping,
            first_mapping_count,
            second_mapping,
            second_mapping_count
        )?;

        let first_labels: &Labels = &*first;
        let second_labels: &Labels = &*second;

        let result_rust = first_labels.intersection(
            second_labels,
            first_mapping,
            second_mapping,
        )?;

        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = mts_labels_t::into_raw(Arc::new(result_rust));

        Ok(())
    })
}

/// Take the difference of two sets of labels.
///
/// If requested, this function can also give the positions in the difference
/// where each entry of the first set of labels ended up.
///
/// This function allocates memory for `*result` which must be released with
/// `mts_labels_free` when you don't need it anymore.
///
/// @param first pointer to the first set of labels
/// @param second pointer to the second set of labels
/// @param result on output, will be set to a pointer to the newly allocated
///        labels containing the difference
/// @param first_mapping if you want the mapping from the positions of entries
///        in `first` to the positions in `result`, this should be a pointer to
///        an array containing `first.count` elements, to be filled by this
///        function. Otherwise it should be a `NULL` pointer. If an entry in
///        `first` is not used in `result`, the mapping will be set to -1.
/// @param first_mapping_count number of elements in the `first_mapping` array
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern "C" fn mts_labels_difference(
    first: *const mts_labels_t,
    second: *const mts_labels_t,
    result: *mut *mut mts_labels_t,
    first_mapping: *mut i64,
    first_mapping_count: usize,
) -> mts_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(result);
    catch_unwind(|| {
        check_pointers_non_null!(first, second);

        let (first_mapping, _) = labels_set_common(
            "difference",
            first,
            second,
            first_mapping,
            first_mapping_count,
            std::ptr::null_mut(),
            0
        )?;

        let first_labels: &Labels = &*first;
        let second_labels: &Labels = &*second;

        let result_rust = first_labels.difference(
            second_labels,
            first_mapping,
        )?;

        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = mts_labels_t::into_raw(Arc::new(result_rust));

        Ok(())
    })
}

/// Select entries in the `labels` that match the `selection`.
///
/// The selection's names must be a subset of the name of the `labels` names.
///
/// All entries in the `labels` that match one of the entry in the `selection`
/// for all the selection's dimension will be picked. Any entry in the
/// `selection` but not in the `labels` will be ignored.
///
/// @param labels pointer to an existing set of labels
/// @param selection pointer to labels defining the selection criteria.
///        Multiple entries are interpreted as a logical `or` operation.
/// @param selected on input, a pointer to an array with space for
///        `*selected_count` entries. On output, the first `*selected_count`
///        values will contain the index in `labels` of selected entries.
/// @param selected_count on input, size of the `selected` array. On output,
///        this will contain the number of selected entries.
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern "C" fn mts_labels_select(
    labels: *const mts_labels_t,
    selection: *const mts_labels_t,
    selected: *mut i64,
    selected_count: *mut usize,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(labels, selection, selected, selected_count);

        let labels_ref: &Labels = &*labels;
        let selection_ref: &Labels = &*selection;

        if *selected_count != labels_ref.count() {
            return Err(Error::InvalidParameter(format!(
                "`selected_count` ({}) must match the number of elements \
                in `labels` ({}) but doesn't",
                *selected_count,
                labels_ref.count(),
            )));
        }

        let selected = std::slice::from_raw_parts_mut(selected, *selected_count);

        *selected_count = labels_ref.select(selection_ref, selected)?;

        Ok(())
    })
}

/// Decrease the reference count of `labels`, and release the corresponding
/// memory once the reference count reaches 0.
///
/// @param labels pointer to an existing set of labels, or NULL
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern "C" fn mts_labels_free(
    labels: *mut mts_labels_t,
) -> mts_status_t {
    catch_unwind(|| {
        if !labels.is_null() {
            std::mem::drop(mts_labels_t::from_raw(labels));
        }

        Ok(())
    })
}
