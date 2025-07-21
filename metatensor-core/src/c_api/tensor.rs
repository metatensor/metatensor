use std::os::raw::c_char;
use std::sync::Arc;
use std::ffi::CStr;
use std::collections::BTreeSet;

use crate::{TensorMap, TensorBlock, Error};

use super::labels::{mts_labels_t, rust_to_mts_labels, mts_labels_to_rust};
use super::blocks::mts_block_t;
use super::status::{mts_status_t, catch_unwind};

/// Opaque type representing a `TensorMap`.
#[allow(non_camel_case_types)]
pub struct mts_tensormap_t(TensorMap);

impl mts_tensormap_t {
    /// Create a raw pointer to `mts_tensormap_t` using a rust Box
    pub fn into_boxed_raw(tensor: TensorMap) -> *mut mts_tensormap_t {
        let boxed = Box::new(mts_tensormap_t(tensor));
        return Box::into_raw(boxed);
    }

    /// Take a raw pointer created by `mts_tensormap_t::into_boxed_raw` and
    /// extract the TensorMap. The pointer is consumed by this function and no
    /// longer valid.
    pub unsafe fn from_boxed_raw(tensor: *mut mts_tensormap_t) -> TensorMap {
        return Box::from_raw(tensor).0;
    }
}

impl std::ops::Deref for mts_tensormap_t {
    type Target = TensorMap;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for mts_tensormap_t {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}


/// Create a new `mts_tensormap_t` with the given `keys` and `blocks`.
/// `blocks_count` must be set to the number of entries in the blocks array.
///
/// The new tensor map takes ownership of the blocks, which should not be
/// released separately.
///
/// The memory allocated by this function and the blocks should be released
/// using `mts_tensormap_free`.
///
/// @param keys labels containing the keys associated with each block
/// @param blocks pointer to the first element of an array of blocks
/// @param blocks_count number of elements in the `blocks` array
///
/// @returns A pointer to the newly allocated tensor map, or a `NULL` pointer in
///          case of error. In case of error, you can use `mts_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern "C" fn mts_tensormap(
    keys: mts_labels_t,
    blocks: *mut *mut mts_block_t,
    blocks_count: usize,
) -> *mut mts_tensormap_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);

    let status = catch_unwind(move || {
        let mut blocks_vec = Vec::new();
        if blocks_count != 0 {
            check_pointers_non_null!(blocks);

            let blocks_slice = std::slice::from_raw_parts_mut(blocks, blocks_count);
            // check for uniqueness of the pointers: we don't want to move out
            // the same value twice
            if blocks_slice.iter().collect::<BTreeSet<_>>().len() != blocks_slice.len() {
                return Err(Error::InvalidParameter(
                    "got the same block more than once when constructing a tensor map".into()
                ));
            }

            for block_ptr in blocks_slice {
                // move out of the blocks pointers
                let block = Box::from_raw(*block_ptr).into_block();
                *block_ptr = std::ptr::null_mut();
                blocks_vec.push(block);
            }
        }

        let keys = mts_labels_to_rust(&keys)?;
        let tensor = TensorMap::new(keys, blocks_vec)?;

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = mts_tensormap_t::into_boxed_raw(tensor);
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}


/// Free the memory associated with a `tensor` previously created with
/// `mts_tensormap`.
///
/// If `tensor` is `NULL`, this function does nothing.
///
/// @param tensor pointer to an existing tensor map, or `NULL`
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern "C" fn mts_tensormap_free(tensor: *mut mts_tensormap_t) -> mts_status_t {
    catch_unwind(|| {
        if !tensor.is_null() {
            std::mem::drop(mts_tensormap_t::from_boxed_raw(tensor));
        }

        Ok(())
    })
}

/// Make a copy of an `mts_tensormap_t`.
///
/// The memory allocated by this function and the blocks should be released
/// using `mts_tensormap_free`.
///
/// @param tensor existing tensor to copy
///
/// @returns A pointer to the newly allocated tensor, or a `NULL` pointer in
///          case of error. In case of error, you can use `mts_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern "C" fn mts_tensormap_copy(
    tensor: *const mts_tensormap_t,
) -> *mut mts_tensormap_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);
    let status = catch_unwind(move || {
        check_pointers_non_null!(tensor);
        let new_tensor = (*tensor).try_clone()?;
        let boxed = Box::new(mts_tensormap_t(new_tensor));

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *(unwind_wrapper.0) = Box::into_raw(boxed);
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}


/// Get the keys for the given `tensor` map.
///
/// This function allocates memory for `keys` which must be released
/// `mts_labels_free` when you don't need it anymore.
///
/// @param tensor pointer to an existing tensor map
/// @param keys pointer to be filled with the keys of the tensor map
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern "C" fn mts_tensormap_keys(
    tensor: *const mts_tensormap_t,
    keys: *mut mts_labels_t,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(tensor, keys);

        if (*keys).is_rust() {
            return Err(Error::InvalidParameter(
                "these labels are already allocated, call mts_labels_free first".into()
            ));
        }

        *keys = rust_to_mts_labels(Arc::clone((*tensor).keys()));
        Ok(())
    })
}


/// Get a pointer to the `index`-th block in this tensor map.
///
/// The block memory is still managed by the tensor map, this block should not
/// be freed. The block is invalidated when the tensor map is freed with
/// `mts_tensormap_free` or the set of keys is modified by calling one
/// of the `mts_tensormap_keys_to_XXX` function.
///
/// @param tensor pointer to an existing tensor map
/// @param block pointer to be filled with a block
/// @param index index of the block to get
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern "C" fn mts_tensormap_block_by_id(
    tensor: *mut mts_tensormap_t,
    block: *mut *mut mts_block_t,
    index: usize,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(tensor, block);

        let blocks = (*tensor).blocks_mut();
        match blocks.get_mut(index) {
            Some(b) => {
                (*block) = (b as *mut TensorBlock).cast();
            }
            None => {
                return Err(Error::InvalidParameter(format!(
                    "block index out of bounds: we have {} blocks but the index is {}",
                    blocks.len(), index
                )));
            }
        }

        Ok(())
    })
}


/// Get indices of the blocks in this `tensor` corresponding to the given
/// `selection`. The `selection` should have a subset of the names/dimensions of
/// the keys for this tensor map, and only one entry, describing the requested
/// blocks.
///
/// When calling this function, `*count` should contain the number of entries in
/// `block_indexes`. When the function returns successfully, `*count` will
/// contain the number of blocks matching the selection, i.e. how many values
/// were written to `block_indexes`.
///
/// @param tensor pointer to an existing tensor map
/// @param block_indexes array to be filled with indexes of blocks in the tensor
///                      map matching the `selection`
/// @param count number of entries in `block_indexes`
/// @param selection labels with a single entry describing which blocks are requested
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern "C" fn mts_tensormap_blocks_matching(
    tensor: *const mts_tensormap_t,
    block_indexes: *mut usize,
	count: *mut usize,
    selection: mts_labels_t,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(tensor, count);

        if *count != (*tensor).keys().count() {
            return Err(Error::InvalidParameter(format!(
                "expected space for {} indices as input to mts_tensormap_blocks_matching, got space for {}",
                (*tensor).keys().count(), *count
            )));
        }

        let selection = mts_labels_to_rust(&selection)?;
        let rust_blocks = (*tensor).blocks_matching(&selection)?;
		*count = rust_blocks.len();

        if (*tensor).keys().is_empty() {
            return Ok(());
        }

        check_pointers_non_null!(block_indexes);
        let block_indexes = std::slice::from_raw_parts_mut(block_indexes, *count);
		for (idx,block) in rust_blocks.into_iter().enumerate() {
            block_indexes[idx] = block;
		}

        Ok(())
    })
}


/// Merge blocks with the same value for selected keys dimensions along the
/// property axis.
///
/// The dimensions (names) of `keys_to_move` will be moved from the keys to
/// the property labels, and blocks with the same remaining keys dimensions
/// will be merged together along the property axis.
///
/// If `keys_to_move` does not contains any entries (`keys_to_move.count
/// == 0`), then the new property labels will contain entries corresponding
/// to the merged blocks only. For example, merging a block with key `a=0`
/// and properties `p=1, 2` with a block with key `a=2` and properties `p=1,
/// 3` will produce a block with properties `a, p = (0, 1), (0, 2), (2, 1),
/// (2, 3)`.
///
/// If `keys_to_move` contains entries, then the property labels must be the
/// same for all the merged blocks. In that case, the merged property labels
/// will contains each of the entries of `keys_to_move` and then the current
/// property labels. For example, using `a=2, 3` in `keys_to_move`, and
/// blocks with properties `p=1, 2` will result in `a, p = (2, 1), (2, 2),
/// (3, 1), (3, 2)`.
///
/// The new sample labels will contains all of the merged blocks sample
/// labels. The order of the samples is controlled by `sort_samples`. If
/// `sort_samples` is true, samples are re-ordered to keep them
/// lexicographically sorted. Otherwise they are kept in the order in which
/// they appear in the blocks.
///
/// The result is a new tensor map, which should be freed with `mts_tensormap_free`.
///
/// @param tensor pointer to an existing tensor map
/// @param keys_to_move description of the keys to move
/// @param sort_samples whether to sort the samples lexicographically after
///                     merging blocks
///
/// @returns A pointer to the newly allocated tensor map, or a `NULL` pointer in
///          case of error. In case of error, you can use `mts_last_error()`
///          to get the error message.
#[no_mangle]
pub unsafe extern "C" fn mts_tensormap_keys_to_properties(
    tensor: *const mts_tensormap_t,
    keys_to_move: mts_labels_t,
    sort_samples: bool,
) -> *mut mts_tensormap_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);

    let status = catch_unwind(move || {
        check_pointers_non_null!(tensor);

        let keys_to_move = mts_labels_to_rust(&keys_to_move)?;
        let moved = (*tensor).keys_to_properties(&keys_to_move, sort_samples)?;

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = mts_tensormap_t::into_boxed_raw(moved);
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}


/// Move the given dimensions from the component labels to the property labels
/// for each block in this tensor map.
///
/// `dimensions` must be an array of `dimensions_count` NULL-terminated strings,
/// encoded as UTF-8.
///
/// @param tensor pointer to an existing tensor map
/// @param dimensions names of the key dimensions to move to the properties
/// @param dimensions_count number of entries in the `dimensions` array
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern "C" fn mts_tensormap_components_to_properties(
    tensor: *mut mts_tensormap_t,
    dimensions: *const *const c_char,
    dimensions_count: usize,
) -> *mut mts_tensormap_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);

    let status = catch_unwind(move || {
        check_pointers_non_null!(tensor, dimensions);

        assert!(dimensions_count != 0);
        let mut rust_dimensions = Vec::new();
        for &dimension in std::slice::from_raw_parts(dimensions, dimensions_count) {
            check_pointers_non_null!(dimension);
            let dimension = CStr::from_ptr(dimension).to_str().expect("invalid utf8");
            rust_dimensions.push(dimension);
        }

        let moved = (*tensor).components_to_properties(&rust_dimensions)?;

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = mts_tensormap_t::into_boxed_raw(moved);

        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}

/// Merge blocks with the same value for selected keys dimensions along the
/// samples axis.
///
/// The dimensions (names) of `keys_to_move` will be moved from the keys to
/// the sample labels, and blocks with the same remaining keys dimensions
/// will be merged together along the sample axis.
///
/// `keys_to_move` must be empty (`keys_to_move.count == 0`), and the new
/// sample labels will contain entries corresponding to the merged blocks'
/// keys.
///
/// The new sample labels will contains all of the merged blocks sample
/// labels. The order of the samples is controlled by `sort_samples`. If
/// `sort_samples` is true, samples are re-ordered to keep them
/// lexicographically sorted. Otherwise they are kept in the order in which
/// they appear in the blocks.
///
/// This function is only implemented if all merged block have the same
/// property labels.
///
/// @param tensor pointer to an existing tensor map
/// @param keys_to_move description of the keys to move
/// @param sort_samples whether to sort the samples lexicographically after
///                     merging blocks or not
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern "C" fn mts_tensormap_keys_to_samples(
    tensor: *const mts_tensormap_t,
    keys_to_move: mts_labels_t,
    sort_samples: bool,
) -> *mut mts_tensormap_t {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);

    let status = catch_unwind(move || {
        check_pointers_non_null!(tensor);

        let keys_to_move = mts_labels_to_rust(&keys_to_move)?;
        let moved = (*tensor).keys_to_samples(&keys_to_move, sort_samples)?;

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = mts_tensormap_t::into_boxed_raw(moved);
        Ok(())
    });

    if !status.is_success() {
        return std::ptr::null_mut();
    }

    return result;
}
