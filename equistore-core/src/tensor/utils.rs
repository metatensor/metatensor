use std::collections::BTreeSet;
use std::sync::Arc;

use indexmap::IndexSet;

use crate::labels::{Labels, LabelsBuilder, LabelValue};
use crate::{Error, TensorBlock, eqs_sample_mapping_t};

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
pub fn remove_dimensions_from_keys(keys: &Labels, dimensions: &[&str]) -> Result<RemovedDimensionsKeys, Error> {
    let names = keys.names();
    for dimension in dimensions {
        if !names.contains(dimension) {
            return Err(Error::InvalidParameter(format!(
                "'{}' is not part of the keys for this tensor map",
                dimension
            )));
        }
    }

    let mut extracted_i = Vec::new();
    for &dimension in dimensions {
        for (i, &name) in names.iter().enumerate() {
            if dimension == name {
                extracted_i.push(i);
            }
        }
    }

    let mut remaining_names = Vec::new();
    let mut remaining_i = Vec::new();
    for (i, &name) in names.iter().enumerate() {
        if !extracted_i.contains(&i) {
            remaining_names.push(name);
            remaining_i.push(i);
        }
    }

    let remaining_keys = if remaining_i.is_empty() {
        let mut builder = LabelsBuilder::new(vec!["_"])?;
        builder.add(&[0])?;
        builder.finish()
    } else {
        let mut remaining_keys = IndexSet::new();
        for key in keys.iter() {
            let mut label = Vec::new();
            for &i in &remaining_i {
                label.push(key[i]);
            }
            remaining_keys.insert(label);
        }

        let mut remaining_keys_builder = LabelsBuilder::new(remaining_names)?;
        for entry in remaining_keys {
            remaining_keys_builder.add(&entry)?;
        }
        remaining_keys_builder.finish()
    };

    return Ok(RemovedDimensionsKeys {
        new_keys: remaining_keys,
        dimensions_positions: extracted_i,
    });
}

pub fn merge_gradient_samples(
    blocks: &[KeyAndBlock],
    gradient_name: &str,
    samples_mappings: &[Vec<eqs_sample_mapping_t>],
) -> Result<Arc<Labels>, Error> {
    let mut new_gradient_samples = BTreeSet::new();
    let mut new_gradient_samples_names = None;
    for (KeyAndBlock{block, ..}, samples_mapping) in blocks.iter().zip(samples_mappings) {
        let gradient = block.gradient(gradient_name).expect("missing gradient");

        if new_gradient_samples_names.is_none() {
            new_gradient_samples_names = Some(gradient.samples.names());
        }

        for grad_sample in gradient.samples.iter() {
            // translate from the old sample id in gradients to the new ones
            let mut grad_sample = grad_sample.to_vec();
            let old_sample_i = grad_sample[0].usize();

            let mapping = &samples_mapping[old_sample_i];
            debug_assert_eq!(mapping.input, old_sample_i);
            grad_sample[0] = mapping.output.into();

            new_gradient_samples.insert(grad_sample);
        }
    }

    let mut new_gradient_samples_builder = LabelsBuilder::new(
        new_gradient_samples_names.expect("missing gradient samples names")
    )?;

    for sample in new_gradient_samples {
        new_gradient_samples_builder.add(&sample)?;
    }

    return Ok(Arc::new(new_gradient_samples_builder.finish()));
}

pub fn merge_samples(
    blocks: &[KeyAndBlock],
    new_sample_names: Vec<&str>,
    sort: bool
) -> (Arc<Labels>, Vec<Vec<eqs_sample_mapping_t>>) {
    let add_key_to_samples = blocks[0].block.samples.size() < new_sample_names.len();

    // Collect samples in an IndexSet to keep them in the same order as they
    // were in the blocks, and then optionally sort them later below
    let mut merged_samples = IndexSet::new();
    for KeyAndBlock{key, block} in blocks {
        for sample in block.samples.iter() {
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

    let mut merged_samples_builder = LabelsBuilder::new(new_sample_names).expect("invalid new samples names");
    for sample in merged_samples {
        merged_samples_builder.add(&sample).expect("got duplicated samples");
    }

    let merged_samples = Arc::new(merged_samples_builder.finish());

    let mut samples_mappings = Vec::new();
    for KeyAndBlock{key, block} in blocks {
        let mut mapping_for_block = Vec::new();
        for (sample_i, sample) in block.samples.iter().enumerate() {
            let mut sample = sample.to_vec();
            if add_key_to_samples {
                sample.extend_from_slice(key);
            }

            let new_sample_i = merged_samples.position(&sample).expect("missing entry in merged samples");
            mapping_for_block.push(eqs_sample_mapping_t {
                input: sample_i,
                output: new_sample_i,
            });
        }
        samples_mappings.push(mapping_for_block);
    }

    return (merged_samples, samples_mappings)
}

/******************************************************************************/

#[cfg(test)]
pub use self::tests_utils::example_labels;

#[cfg(test)]
mod tests_utils {
    use std::sync::Arc;
    use crate::labels::{Labels, LabelsBuilder, LabelValue};

    pub fn example_labels<const N: usize>(names: Vec<&str>, values: Vec<[i32; N]>) -> Arc<Labels> {
        let mut labels = LabelsBuilder::new(names).unwrap();
        for entry in values {
            labels.add(
                &entry.iter().copied().map(LabelValue::from).collect::<Vec<_>>()
            ).unwrap();
        }
        return Arc::new(labels.finish());
    }
}
