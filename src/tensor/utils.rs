use std::collections::BTreeSet;
use std::sync::Arc;

use indexmap::IndexSet;

use crate::labels::{Labels, LabelsBuilder, LabelValue};
use crate::{Error, TensorBlock, eqs_sample_mapping_t};

/// single block and part of the associated key, this is used for the various
/// `keys_to_xxx` functions
pub type KeyAndBlock<'a> = (Vec<LabelValue>, &'a TensorBlock);

/// Result of the `remove_variables_from_keys` function
#[derive(Debug, Clone)]
pub struct RemovedVariablesKeys {
    /// keys without the variables
    pub(super) new_keys: Labels,
    /// positions of the moved variables in the original keys
    pub(super) variables_positions: Vec<usize>,
}

/// Remove the given variables from these keys, returning the updated set of
/// keys and the positions of the removed variables in the initial keys
pub fn remove_variables_from_keys(keys: &Labels, variables: &[&str]) -> Result<RemovedVariablesKeys, Error> {
    let names = keys.names();
    for variable in variables {
        if !names.contains(variable) {
            return Err(Error::InvalidParameter(format!(
                "'{}' is not part of the keys for this tensor map",
                variable
            )));
        }
    }

    let mut extracted_i = Vec::new();
    for &variable in variables {
        for (i, &name) in names.iter().enumerate() {
            if variable == name {
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
        Labels::single()
    } else {
        let mut remaining_keys = IndexSet::new();
        for key in keys.iter() {
            let mut label = Vec::new();
            for &i in &remaining_i {
                label.push(key[i]);
            }
            remaining_keys.insert(label);
        }

        let mut remaining_keys_builder = LabelsBuilder::new(remaining_names);
        for entry in remaining_keys {
            remaining_keys_builder.add(&entry);
        }
        remaining_keys_builder.finish()
    };

    return Ok(RemovedVariablesKeys {
        new_keys: remaining_keys,
        variables_positions: extracted_i,
    });
}

pub fn merge_gradient_samples(
    blocks: &[KeyAndBlock],
    gradient_name: &str,
    samples_mappings: &[Vec<eqs_sample_mapping_t>],
) -> Arc<Labels> {
    let mut new_gradient_samples = BTreeSet::new();
    let mut new_gradient_samples_names = None;
    for ((_, block), samples_mapping) in blocks.iter().zip(samples_mappings) {
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

    let mut new_gradient_samples_builder = LabelsBuilder::new(new_gradient_samples_names.expect("missing gradient samples names"));
    for sample in new_gradient_samples {
        new_gradient_samples_builder.add(&sample);
    }

    return Arc::new(new_gradient_samples_builder.finish());
}

pub fn merge_samples(
    blocks: &[KeyAndBlock],
    new_sample_names: Vec<&str>,
    sort: bool
) -> (Arc<Labels>, Vec<Vec<eqs_sample_mapping_t>>) {
    let add_key_to_samples = blocks[0].1.values().samples.size() < new_sample_names.len();

    // Collect samples in an IndexSet to keep them in the same order as they
    // were in the blocks, and then optionally sort them later below
    let mut merged_samples = IndexSet::new();
    for (key, block) in blocks {
        for sample in block.values().samples.iter() {
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

    let mut merged_samples_builder = LabelsBuilder::new(new_sample_names);
    for sample in merged_samples {
        merged_samples_builder.add(&sample);
    }

    let merged_samples = Arc::new(merged_samples_builder.finish());

    let mut samples_mappings = Vec::new();
    for (key, block) in blocks {
        let mut mapping_for_block = Vec::new();
        for (sample_i, sample) in block.values().samples.iter().enumerate() {
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

#[cfg(all(test, feature = "ndarray"))]
pub use self::tests_utils_ndarray::{example_block, example_tensor};

#[cfg(test)]
pub use self::tests_utils::example_labels;

#[cfg(test)]
mod tests_utils {
    use std::sync::Arc;
    use crate::labels::{Labels, LabelsBuilder, LabelValue};

    pub fn example_labels<const N: usize>(names: Vec<&str>, values: Vec<[i32; N]>) -> Arc<Labels> {
        let mut labels = LabelsBuilder::new(names);
        for entry in values {
            labels.add(&entry.iter().copied().map(LabelValue::from).collect::<Vec<_>>());
        }
        return Arc::new(labels.finish());
    }
}


#[cfg(all(test, feature = "ndarray"))]
mod tests_utils_ndarray {
    use super::tests_utils::example_labels;

    use crate::{LabelsBuilder, TensorBlock, TensorMap};

    use ndarray::ArrayD;

    pub fn example_block(
        samples: Vec<[i32; 1]>,
        components: Vec<[i32; 1]>,
        properties: Vec<[i32; 1]>,
        gradient_samples: Vec<[i32; 2]>,
        values: f64,
        gradient_values: f64
    ) -> TensorBlock {
        let samples = example_labels(vec!["samples"], samples);
        let components = example_labels(vec!["components"], components);
        let properties = example_labels(vec!["properties"], properties);

        let shape = vec![samples.count(), components.count(), properties.count()];
        let mut block = TensorBlock::new(
            ArrayD::from_elem(shape, values),
            samples,
            vec![components.clone()],
            properties.clone(),
        ).unwrap();

        let gradient_samples = example_labels(vec!["sample", "parameter"], gradient_samples);

        let shape = vec![gradient_samples.count(), components.count(), properties.count()];
        block.add_gradient(
            "parameter",
            ArrayD::from_elem(shape, gradient_values),
            gradient_samples,
            vec![components],
        ).unwrap();

        return block;
    }

    pub fn example_tensor() -> TensorMap {
        let block_1 = example_block(
            /* samples          */ vec![[0], [2], [4]],
            /* components       */ vec![[0]],
            /* properties       */ vec![[0]],
            /* gradient_samples */ vec![[0, -2], [2, 3]],
            /* values           */ 1.0,
            /* gradient_values  */ 11.0,
        );

        // different property size
        let block_2 = example_block(
            /* samples          */ vec![[0], [1], [3]],
            /* components       */ vec![[0]],
            /* properties       */ vec![[3], [4], [5]],
            /* gradient_samples */ vec![[0, -2], [0, 3], [2, -2]],
            /* values           */ 2.0,
            /* gradient_values  */ 12.0,
        );

        // different component size
        let block_3 = example_block(
            /* samples          */ vec![[0], [3], [6], [8]],
            /* components       */ vec![[0], [1], [2]],
            /* properties       */ vec![[0]],
            /* gradient_samples */ vec![[1, -2]],
            /* values           */ 3.0,
            /* gradient_values  */ 13.0,
        );

        let block_4 = example_block(
            /* samples          */ vec![[0], [1], [2], [5]],
            /* components       */ vec![[0], [1], [2]],
            /* properties       */ vec![[0]],
            /* gradient_samples */ vec![[0, 1], [3, 3]],
            /* values           */ 4.0,
            /* gradient_values  */ 14.0,
        );

        let mut keys = LabelsBuilder::new(vec!["key_1", "key_2"]);
        keys.add(&[0, 0]);
        keys.add(&[1, 0]);
        keys.add(&[2, 2]);
        keys.add(&[2, 3]);
        let keys = keys.finish();

        return TensorMap::new(keys, vec![block_1, block_2, block_3, block_4]).unwrap();
    }
}
