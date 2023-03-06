use std::os::raw::{c_char, c_void};
use std::ffi::CStr;
use std::sync::Arc;

use crate::{LabelValue, Labels, LabelsBuilder, Error};
use super::status::{eqs_status_t, catch_unwind};

/// A set of labels used to carry metadata associated with a tensor map.
///
/// This is similar to a list of `count` named tuples, but stored as a 2D array
/// of shape `(count, size)`, with a set of names associated with the columns of
/// this array (often called *dimensions*). Each row/entry in this array is
/// unique, and they are often (but not always) sorted in lexicographic order.
///
/// `eqs_labels_t` with a non-NULL `internal_ptr_` correspond to a
/// reference-counted Rust data structure, which allow for fast lookup inside
/// the labels with `eqs_labels_positions`.

// An `eqs_labels_t` can either correspond to a Rust `Arc<Labels>` (`labels_ptr`
// is non-NULL, and corresponds to the pointer `Arc::into_raw` gives); or to a
// set of Labels created from C, and containing pointer to C-allocated data.
#[repr(C)]
pub struct eqs_labels_t {
    /// internal: pointer to the rust `Labels` struct if any, null otherwise
    pub internal_ptr_: *mut c_void,

    /// Names of the dimensions composing this set of labels. There are `size`
    /// elements in this array, each being a NULL terminated UTF-8 string.
    pub names: *const *const c_char,
    /// Pointer to the first element of a 2D row-major array of 32-bit signed
    /// integer containing the values taken by the different dimensions in
    /// `names`. Each row has `size` elements, and there are `count` rows in
    /// total.
    pub values: *const i32,
    /// Number of dimensions/size of a single entry in the set of labels
    pub size: usize,
    /// Number entries in the set of labels
    pub count: usize,
}

impl eqs_labels_t {
    /// Check if these eqs_labels_t are associated with a Rust `Arc<Labels>` or
    /// not
    pub fn is_rust(&self) -> bool {
        !self.internal_ptr_.is_null()
    }
}

/// Create a new `eqs_labels_t` from a Rust `Arc<Labels>`
pub unsafe fn rust_to_eqs_labels(labels: Arc<Labels>) -> eqs_labels_t {
    let size = labels.size();
    let count = labels.count();

    let values = if labels.count() == 0 || labels.size() == 0 {
        std::ptr::null()
    } else {
        (&labels[0][0] as *const LabelValue).cast()
    };

    let names = if labels.size() == 0 {
        std::ptr::null()
    } else {
        labels.c_names().as_ptr().cast()
    };

    let internal_ptr_ = Arc::into_raw(labels).cast::<c_void>() as *mut c_void;

    eqs_labels_t {
        internal_ptr_,
        names,
        values,
        size,
        count
    }
}

/// Convert from `eqs_label_t` back to a Rust `Arc<Labels>`, potentially
/// constructing new `Labels` if they don't exist yet.
pub unsafe fn eqs_labels_to_rust(labels: &eqs_labels_t) -> Result<Arc<Labels>, Error> {
    // if the labels have already been constructed on the rust side,
    // increase the reference count of the arc & return that
    if labels.is_rust() {
        let labels = Arc::from_raw(labels.internal_ptr_.cast());
        let cloned = Arc::clone(&labels);

        // keep the original arc alive
        std::mem::forget(labels);

        return Ok(cloned);
    }

    // otherwise, create new labels from the data
    return create_rust_labels(labels);
}

/// Create a new set of rust Labels from `eqs_labels_t`, copying the data into
/// Rust managed memory.
unsafe fn create_rust_labels(labels: &eqs_labels_t) -> Result<Arc<Labels>, Error> {
    assert!(!labels.is_rust());

    if labels.names.is_null() {
        return Err(Error::InvalidParameter("labels.names can not be NULL in eqs_labels_t".into()))
    }

    if labels.values.is_null() && labels.count > 0 {
        return Err(Error::InvalidParameter("labels.values is NULL but labels.count is >0 in eqs_labels_t".into()))
    }

    let mut names = Vec::new();
    for i in 0..labels.size {
        let name = CStr::from_ptr(*(labels.names.add(i)));
        let name = name.to_str().expect("invalid UTF8 name");
        if !crate::labels::is_valid_label_name(name) {
            return Err(Error::InvalidParameter(format!(
                "'{}' is not a valid label name", name
            )));
        }
        names.push(name);
    }

    let mut builder = LabelsBuilder::new(names);
    builder.reserve(labels.count);

    let slice = std::slice::from_raw_parts(labels.values.cast::<LabelValue>(), labels.count * labels.size);
    if !slice.is_empty() {
        for chunk in slice.chunks_exact(labels.size) {
            builder.add(chunk)?;
        }
    }

    return Ok(Arc::new(builder.finish()));
}


/// Get the position of the entry defined by the `values` array in the given set
/// of `labels`. This operation is only available if the labels correspond to a
/// set of Rust Labels (i.e. `labels.internal_ptr_` is not NULL).
///
/// @param labels set of labels with an associated Rust data structure
/// @param values array containing the label to lookup
/// @param values_count size of the values array
/// @param result position of the values in the labels or -1 if the values
///               were not found
///
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_labels_position(
    labels: eqs_labels_t,
    values: *const i32,
    values_count: usize,
    result: *mut i64
) -> eqs_status_t {
    catch_unwind(|| {
        if !labels.is_rust() {
            return Err(Error::InvalidParameter(
                "these labels do not support calling eqs_labels_position, \
                call eqs_labels_create first".into()
            ));
        }

        let labels = &(*labels.internal_ptr_.cast::<Labels>());
        if values_count != labels.size() {
            return Err(Error::InvalidParameter(format!(
                "expected label of size {} in eqs_labels_position, got size {}",
                (*labels).size(), values_count
            )));
        }

        let label = std::slice::from_raw_parts(values.cast(), values_count);
        *result = labels.position(label).map_or(-1, |p| p as i64);

        Ok(())
    })
}


/// Finish the creation of `eqs_labels_t` by associating it to Rust-owned
/// labels.
///
/// This allows using the `eqs_labels_positions` and `eqs_labels_clone`
/// functions on the `eqs_labels_t`.
///
/// This function allocates memory which must be released `eqs_labels_free` when
/// you don't need it anymore.
///
/// @param labels new set of labels containing pointers to user-managed memory
///        on input, and pointers to Rust-managed memory on output.
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_labels_create(
    labels: *mut eqs_labels_t,
) -> eqs_status_t {
    catch_unwind(|| {
        check_pointers!(labels);

        if (*labels).is_rust() {
            return Err(Error::InvalidParameter(
                "these labels already correspond to rust labels".into()
            ));
        }

        let rust_labels = create_rust_labels(&*labels)?;
        *labels = rust_to_eqs_labels(rust_labels);

        Ok(())
    })
}

/// Make a copy of `labels` inside `clone`.
///
/// Since `eqs_labels_t` are immutable, the copy is actually just a reference
/// count increase, and as such should not be an expensive operation.
///
/// `eqs_labels_free` must be used with `clone` to decrease the reference count
/// and release the memory when you don't need it anymore.
///
/// @param labels set of labels with an associated Rust data structure
/// @param clone empty labels, on output will contain a copy of `labels`
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_labels_clone(
    labels: eqs_labels_t,
    clone: *mut eqs_labels_t,
) -> eqs_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(clone);
    catch_unwind(|| {
        if !labels.is_rust() {
            return Err(Error::InvalidParameter(
                "these labels do not support calling eqs_labels_clone, \
                call eqs_labels_create first".into()
            ));
        }

        if (*clone).is_rust() {
            return Err(Error::InvalidParameter(
                "output labels already contain some data".into()
            ));
        }

        let rust_labels = Arc::from_raw(labels.internal_ptr_.cast::<Labels>());

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = rust_to_eqs_labels(Arc::clone(&rust_labels));

        // keep the original arc alive
        std::mem::forget(rust_labels);

        Ok(())
    })
}

/// Decrease the reference count of `labels`, and release the corresponding
/// memory once the reference count reaches 0.
///
/// @param labels set of labels with an associated Rust data structure
/// @returns The status code of this operation. If the status is not
///          `EQS_SUCCESS`, you can use `eqs_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn eqs_labels_free(
    labels: *mut eqs_labels_t,
) -> eqs_status_t {
    catch_unwind(|| {
        if labels.is_null() {
            return Ok(());
        }

        if !(*labels).is_rust() {
            return Ok(());
        }

        std::mem::drop(Arc::from_raw((*labels).internal_ptr_.cast::<Labels>()));

        (*labels).internal_ptr_ = std::ptr::null_mut();

        Ok(())
    })
}
