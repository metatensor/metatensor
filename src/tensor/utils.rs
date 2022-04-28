use std::collections::BTreeSet;

use indexmap::IndexSet;

use crate::labels::{Labels, LabelsBuilder, LabelValue};
use crate::{Error, TensorBlock, eqs_sample_mapping_t};

/// single block and part of the associated key, this is used for the various
/// `keys_to_xxx` functions
pub type KeyAndBlock<'a> = (Vec<LabelValue>, &'a TensorBlock);

// result of the split_keys function
pub struct SplittedKeys {
    /// keys without the variables
    pub(super) new_keys: Labels,
    /// values taken by the variables in the original keys
    pub(super) extracted_keys: Labels,
    /// positions of the variables in the original keys
    pub(super) extracted_positions: Vec<usize>,
}

/// Split the `keys` into a new set of label without the `variables`; and Labels
/// containing the values taken by `variables`
pub fn split_keys(keys: &Labels, variables: &[&str]) -> Result<SplittedKeys, Error> {
    let names = keys.names();
    for variable in variables {
        if !names.contains(variable) {
            return Err(Error::InvalidParameter(format!(
                "'{}' is not part of the keys for this tensor map",
                variable
            )));
        }
    }

    // TODO: use Labels instead of Vec<&str> for variables to ensure
    // uniqueness of variables names & pass 'requested' values around

    let mut remaining = Vec::new();
    let mut remaining_i = Vec::new();
    let mut extracted_i = Vec::new();

    'outer: for (i, &name) in names.iter().enumerate() {
        for &variable in variables {
            if variable == name {
                extracted_i.push(i);
                continue 'outer;
            }
        }
        remaining.push(name);
        remaining_i.push(i);
    }

    let mut extracted_keys = IndexSet::new();
    let mut remaining_keys = IndexSet::new();
    for key in keys.iter() {
        let mut label = Vec::new();
        for &i in &extracted_i {
            label.push(key[i]);
        }
        extracted_keys.insert(label);

        if !remaining_i.is_empty() {
            let mut label = Vec::new();
            for &i in &remaining_i {
                label.push(key[i]);
            }
            remaining_keys.insert(label);
        }
    }

    let remaining_keys = if remaining_keys.is_empty() {
        Labels::single()
    } else {
        let mut remaining_keys_builder = LabelsBuilder::new(remaining);
        for entry in remaining_keys {
            remaining_keys_builder.add(entry);
        }
        remaining_keys_builder.finish()
    };

    assert!(!extracted_keys.is_empty());
    let mut extracted_keys_builder = LabelsBuilder::new(variables.to_vec());
    for entry in extracted_keys {
        extracted_keys_builder.add(entry);
    }

    return Ok(SplittedKeys {
        new_keys: remaining_keys,
        extracted_keys: extracted_keys_builder.finish(),
        extracted_positions: extracted_i,
    });
}

pub fn merge_gradient_samples(
    blocks: &[KeyAndBlock],
    gradient_name: &str,
    samples_mappings: &[Vec<eqs_sample_mapping_t>],
) -> Labels {
    let mut new_gradient_samples = BTreeSet::new();
    let mut new_gradient_samples_names = None;
    for ((_, block), samples_mapping) in blocks.iter().zip(samples_mappings) {
        let gradient = block.get_gradient(gradient_name).expect("missing gradient");

        if new_gradient_samples_names.is_none() {
            new_gradient_samples_names = Some(gradient.samples().names());
        }

        for grad_sample in gradient.samples().iter() {
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
        new_gradient_samples_builder.add(sample);
    }

    return new_gradient_samples_builder.finish();
}

pub fn merge_samples(
    blocks: &[KeyAndBlock],
    new_sample_names: Vec<&str>,
    sort: bool
) -> (Labels, Vec<Vec<eqs_sample_mapping_t>>) {
    let add_key_to_samples = blocks[0].1.values.samples().size() < new_sample_names.len();

    // Collect samples in an IndexSet to keep them in the same order as they
    // were in the blocks, and then optionally sort them later below
    let mut merged_samples = IndexSet::new();
    for (key, block) in blocks {
        for sample in block.values.samples().iter() {
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
        merged_samples_builder.add(sample);
    }

    let merged_samples = merged_samples_builder.finish();

    let mut samples_mappings = Vec::new();
    for (key, block) in blocks {
        let mut mapping_for_block = Vec::new();
        for (sample_i, sample) in block.values.samples().iter().enumerate() {
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
pub use self::tests_utils::example_tensor;

#[cfg(all(test, feature = "ndarray"))]
mod tests_utils {
    use std::sync::Arc;

    use crate::labels::{Labels, LabelsBuilder, LabelValue};
    use crate::{TensorBlock, TensorMap};
    use crate::eqs_array_t;

    use ndarray::ArrayD;

    fn example_labels(name: &str, values: Vec<i32>) -> Labels {
        let mut labels = LabelsBuilder::new(vec![name]);
        for i in values {
            labels.add(vec![LabelValue::from(i)]);
        }
        return labels.finish();
    }

    pub fn example_tensor() -> TensorMap {
        let mut block_1 = TensorBlock::new(
            eqs_array_t::new(Box::new(ArrayD::from_elem(vec![3, 1, 1], 1.0))),
            example_labels("samples", vec![0, 2, 4]),
            vec![Arc::new(example_labels("components", vec![0]))],
            Arc::new(example_labels("properties", vec![0])),
        ).unwrap();

        let mut gradient_samples_1 = LabelsBuilder::new(vec!["sample", "parameter"]);
        gradient_samples_1.add(vec![LabelValue::new(0), LabelValue::new(-2)]);
        gradient_samples_1.add(vec![LabelValue::new(2), LabelValue::new(3)]);
        let gradient_samples_1 = gradient_samples_1.finish();

        block_1.add_gradient(
            "parameter",
            eqs_array_t::new(Box::new(ArrayD::from_elem(vec![2, 1, 1], 11.0))),
            gradient_samples_1,
            vec![Arc::new(example_labels("components", vec![0]))],
        ).unwrap();

        /******************************************************************/

        let mut block_2 = TensorBlock::new(
            eqs_array_t::new(Box::new(ArrayD::from_elem(vec![3, 1, 3], 2.0))),
            example_labels("samples", vec![0, 1, 3]),
            vec![Arc::new(example_labels("components", vec![0]))],
            // different property size
            Arc::new(example_labels("properties", vec![3, 4, 5])),
        ).unwrap();

        let mut gradient_samples_2 = LabelsBuilder::new(vec!["sample", "parameter"]);
        gradient_samples_2.add(vec![LabelValue::new(0), LabelValue::new(-2)]);
        gradient_samples_2.add(vec![LabelValue::new(0), LabelValue::new(3)]);
        gradient_samples_2.add(vec![LabelValue::new(2), LabelValue::new(-2)]);
        let gradient_samples_2 = gradient_samples_2.finish();

        block_2.add_gradient(
            "parameter",
            eqs_array_t::new(Box::new(ArrayD::from_elem(vec![3, 1, 3], 12.0))),
            gradient_samples_2,
            vec![Arc::new(example_labels("components", vec![0]))],
        ).unwrap();

        /******************************************************************/

        let mut block_3 = TensorBlock::new(
            eqs_array_t::new(Box::new(ArrayD::from_elem(vec![4, 3, 1], 3.0))),
            example_labels("samples", vec![0, 3, 6, 8]),
            // different component size
            vec![Arc::new(example_labels("components", vec![0, 1, 2]))],
            Arc::new(example_labels("properties", vec![0])),
        ).unwrap();

        let mut gradient_samples_3 = LabelsBuilder::new(vec!["sample", "parameter"]);
        gradient_samples_3.add(vec![LabelValue::new(1), LabelValue::new(-2)]);
        let gradient_samples_3 = gradient_samples_3.finish();

        block_3.add_gradient(
            "parameter",
            eqs_array_t::new(Box::new(ArrayD::from_elem(vec![1, 3, 1], 13.0))),
            gradient_samples_3,
            vec![Arc::new(example_labels("components", vec![0, 1, 2]))],
        ).unwrap();

        /******************************************************************/

        let mut block_4 = TensorBlock::new(
            eqs_array_t::new(Box::new(ArrayD::from_elem(vec![4, 3, 1], 4.0))),
            example_labels("samples", vec![0, 1, 2, 5]),
            vec![Arc::new(example_labels("components", vec![0, 1, 2]))],
            Arc::new(example_labels("properties", vec![0])),
        ).unwrap();

        let mut gradient_samples_4 = LabelsBuilder::new(vec!["sample", "parameter"]);
        gradient_samples_4.add(vec![LabelValue::new(0), LabelValue::new(1)]);
        gradient_samples_4.add(vec![LabelValue::new(3), LabelValue::new(3)]);
        let gradient_samples_4 = gradient_samples_4.finish();

        block_4.add_gradient(
            "parameter",
            eqs_array_t::new(Box::new(ArrayD::from_elem(vec![2, 3, 1], 14.0))),
            gradient_samples_4,
            vec![Arc::new(example_labels("components", vec![0, 1, 2]))],
        ).unwrap();

        /******************************************************************/

        let mut keys = LabelsBuilder::new(vec!["key_1", "key_2"]);
        keys.add(vec![LabelValue::new(0), LabelValue::new(0)]);
        keys.add(vec![LabelValue::new(1), LabelValue::new(0)]);
        keys.add(vec![LabelValue::new(2), LabelValue::new(2)]);
        keys.add(vec![LabelValue::new(2), LabelValue::new(3)]);
        let keys = keys.finish();

        return TensorMap::new(keys, vec![block_1, block_2, block_3, block_4]).unwrap();
    }
}
