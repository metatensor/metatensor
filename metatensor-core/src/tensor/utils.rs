use std::collections::BTreeSet;
use std::sync::Arc;

use indexmap::IndexSet;

use crate::labels::{Labels, LabelValue};
use crate::{Error, TensorBlock};

/// single block and part of the associated key, this is used for the various
/// `keys_to_xxx` functions
pub struct KeyAndBlock<'a> {
    pub key: Vec<LabelValue>,
    pub block: &'a TensorBlock,
}

/// Result of the `remove_dimensions_from_keys` function
#[derive(Debug)]
pub struct RemovedDimensionsKeys {
    /// keys without the dimensions
    pub(super) new_keys: Labels,
    /// positions of the moved dimensions in the original keys
    pub(super) dimensions_positions: Vec<usize>,
}

/// Remove the given dimensions from these keys, returning the updated set of
/// keys and the positions of the removed dimensions in the initial keys
pub fn remove_dimensions_from_keys(keys: &Labels, to_remove: &[&str]) -> Result<RemovedDimensionsKeys, Error> {
    let dimensions = keys.dimensions();
    for remove in to_remove {
        if !dimensions.contains(remove) {
            return Err(Error::InvalidParameter(format!(
                "'{}' is not part of the keys for this tensor map",
                remove
            )));
        }
    }

    let mut extracted_i = Vec::new();
    for &remove in to_remove {
        for (i, &dimension) in dimensions.iter().enumerate() {
            if remove == dimension {
                extracted_i.push(i);
            }
        }
    }

    let mut remaining_names = Vec::new();
    let mut remaining_i = Vec::new();
    for (i, &dimension) in dimensions.iter().enumerate() {
        if !extracted_i.contains(&i) {
            remaining_names.push(dimension);
            remaining_i.push(i);
        }
    }

    let mut remaining_keys = IndexSet::new();
    for key in &keys.to_cpu() {
        let mut entry = Vec::new();
        for &i in &remaining_i {
            entry.push(key[i]);
        }
        remaining_keys.insert(entry);
    }

    let values = remaining_keys.into_iter().reduce(|mut values, entry| {
        values.extend_from_slice(&entry);
        values
    }).expect("the set should contain at least an empty vector");

    let remaining_keys = if values.is_empty() {
        Labels::from_vec(&["_"], vec![0]).expect("invalid labels")
    } else {
        unsafe {
            // SAFETY: the values come from an IndexSet and should already be unique
            Labels::from_vec_unchecked_uniqueness(&remaining_names, values).expect("invalid labels")
        }
    };

    return Ok(RemovedDimensionsKeys {
        new_keys: remaining_keys,
        dimensions_positions: extracted_i,
    });
}

pub fn merge_gradient_samples(
    blocks: &[KeyAndBlock],
    gradient_name: &str,
    samples_mappings: &[Vec<usize>],
) -> Arc<Labels> {
    let mut new_sample_values = BTreeSet::new();
    let mut new_sample_dimensions = None;
    for (KeyAndBlock{block, ..}, samples_mapping) in blocks.iter().zip(samples_mappings) {
        let gradient = block.gradient(gradient_name).expect("missing gradient");

        if new_sample_dimensions.is_none() {
            new_sample_dimensions = Some(gradient.samples.dimensions());
        }

        for grad_sample in &gradient.samples.to_cpu() {
            // translate from the old sample id in gradients to the new ones
            let mut grad_sample = grad_sample.to_vec();
            let old_sample_i = usize::try_from(grad_sample[0]).expect("could not convert to usize");

            let new_sample_i = samples_mapping[old_sample_i];
            grad_sample[0] = i32::try_from(new_sample_i).expect("could not convert to i32");

            new_sample_values.insert(grad_sample);
        }
    }

    let labels = unsafe {
        // SAFETY: values should already be unique since they come from a set
        Labels::from_vec_unchecked_uniqueness(
            &new_sample_dimensions.expect("missing gradient sample names"),
            new_sample_values.iter().flatten().copied().collect()
        ).expect("invalid labels")
    };

    return Arc::new(labels);
}

pub fn merge_samples(
    blocks: &[KeyAndBlock],
    new_sample_names: &[&str],
    sort: bool
) -> (Arc<Labels>, Vec<Vec<usize>>) {
    let add_key_to_samples = blocks[0].block.samples.size() < new_sample_names.len();

    // Collect samples in an IndexSet to keep them in the same order as they
    // were in the blocks, and then optionally sort them later below
    let mut merged_samples = IndexSet::new();
    for KeyAndBlock{key, block} in blocks {
        for sample in &block.samples.to_cpu() {
            let mut sample = sample.to_vec();
            if add_key_to_samples {
                sample.extend_from_slice(key);
            }

            merged_samples.insert(sample);
        }
    }

    if sort {
        merged_samples.sort_unstable();
    }

    let merged_samples = unsafe {
        // SAFETY: values should already be unique since they come from a set
        Labels::from_vec_unchecked_uniqueness(
            new_sample_names,
            merged_samples.iter().flatten().copied().collect()
        ).expect("invalid labels")
    };

    let merged_samples = Arc::new(merged_samples);

    let mut samples_mappings = Vec::new();
    for KeyAndBlock{key, block} in blocks {
        let mut mapping_for_block = Vec::with_capacity(block.samples.count());

        for sample in &block.samples.to_cpu() {
            let mut sample = sample.to_vec();
            if add_key_to_samples {
                sample.extend_from_slice(key);
            }

            let new_sample_i = merged_samples.position(&sample).expect("missing entry in merged samples");
            mapping_for_block.push(new_sample_i);
        }
        samples_mappings.push(mapping_for_block);
    }

    return (merged_samples, samples_mappings);
}

/******************************************************************************/

#[cfg(test)]
pub use self::tests_utils::example_labels;

#[cfg(test)]
mod tests_utils {
    use std::sync::Arc;
    use crate::labels::Labels;

    pub fn example_labels(names: &[&str], values: &[i32]) -> Arc<Labels> {
        return Arc::new(Labels::from_vec(names, values.to_vec()).expect("invalid labels"));
    }
}
