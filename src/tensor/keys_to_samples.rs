use std::sync::Arc;

use crate::labels::{Labels, LabelsBuilder, LabelValue};
use crate::{Error, TensorBlock};

use crate::data::eqs_sample_move_t;

use super::TensorMap;
use super::utils::{KeyAndBlock, split_keys, merge_samples, merge_gradient_samples};

impl TensorMap {
    /// Move the given variables from the keys to the sample labels of the
    /// blocks.
    ///
    /// Blocks containing the same values in the keys for the `variables` will
    /// be merged together. The resulting merged blocks will have `variables` as
    /// the last sample variables, preceded by the current samples.
    ///
    /// This function is only implemented if all merged block have the same
    /// property labels.
    ///
    /// The order of the samples is controlled by `sort_samples`. If
    /// `sort_samples` is true, samples are re-ordered to keep them
    /// lexicographically sorted. Otherwise they are kept in the order in which
    /// they appear in the blocks.
    pub fn keys_to_samples(&mut self, variables: &[&str], sort_samples: bool) -> Result<(), Error> {
        // TODO: requested values
        // TODO: keys_to_samples_no_gradients?

        if variables.is_empty() {
            return Ok(());
        }

        let splitted_keys = split_keys(&self.keys, variables)?;

        let mut new_blocks = Vec::new();
        if splitted_keys.new_keys.count() == 1 {
            // create a single block with everything
            let blocks_to_merge = self.keys.iter()
                .zip(&self.blocks)
                .map(|(key, block)| {
                    let mut extracted = Vec::new();
                    for &i in &splitted_keys.extracted_positions {
                        extracted.push(key[i]);
                    }

                    (extracted, block)
                })
                .collect::<Vec<_>>();

            let block = merge_blocks_along_samples(
                &blocks_to_merge, &splitted_keys.extracted_keys, sort_samples
            )?;
            new_blocks.push(block);
        } else {
            for entry in splitted_keys.new_keys.iter() {
                let mut selection = LabelsBuilder::new(splitted_keys.new_keys.names());
                selection.add(entry.to_vec());

                let matching = self.find_matching_blocks(&selection.finish())?;
                let blocks_to_merge = matching.iter()
                    .map(|&i| {
                        let block = &self.blocks[i];
                        let key = &self.keys[i];
                        let mut extracted = Vec::new();
                        for &i in &splitted_keys.extracted_positions {
                            extracted.push(key[i]);
                        }

                        (extracted, block)
                    })
                    .collect::<Vec<_>>();

                new_blocks.push(merge_blocks_along_samples(
                    &blocks_to_merge, &splitted_keys.extracted_keys, sort_samples
                )?);
            }
        }


        self.keys = splitted_keys.new_keys;
        self.blocks = new_blocks;

        return Ok(());
    }
}

/// Merge the given `blocks` along the sample axis.
fn merge_blocks_along_samples(
    blocks_to_merge: &[KeyAndBlock],
    extracted_samples: &Labels,
    sort_samples: bool,
) -> Result<TensorBlock, Error> {
    assert!(!blocks_to_merge.is_empty());

    let first_block = blocks_to_merge[0].1;
    let first_components_label = first_block.values.components();
    let first_properties_label = first_block.values.properties();

    for (_, block) in blocks_to_merge {
        if block.values.components() != first_components_label {
            return Err(Error::InvalidParameter(
                "can not move keys to samples if the blocks have \
                different components labels, call components_to_properties first".into()
            ))
        }

        if block.values.properties() != first_properties_label {
            return Err(Error::InvalidParameter(
                "can not move keys to samples if the blocks have \
                different property labels".into() // TODO: this might be possible
            ))
        }
    }

    // collect and merge samples across the blocks
    let new_samples_names = first_block.values.samples().names().iter().copied()
        .chain(extracted_samples.names())
        .collect();
    let (merged_samples, samples_mapping) = merge_samples(
        blocks_to_merge,
        new_samples_names,
        sort_samples,
    );

    let new_components = first_block.values.components().to_vec();
    let new_properties = Arc::clone(first_block.values.properties());

    let mut new_shape = first_block.values.data.shape()?.to_vec();
    new_shape[0] = merged_samples.count();
    let mut new_data = first_block.values.data.create(&new_shape)?;

    let property_range = 0..new_properties.count();

    debug_assert_eq!(blocks_to_merge.len(), samples_mapping.len());
    for ((_, block), samples_mapping) in blocks_to_merge.iter().zip(&samples_mapping) {
        let mut samples_to_move = Vec::new();
        for (sample_i, &new_sample_i) in samples_mapping.iter().enumerate() {
            samples_to_move.push(eqs_sample_move_t {
                input: sample_i,
                output: new_sample_i,
            });
        }
        new_data.move_samples_from(
            &block.values.data,
            &samples_to_move,
            property_range.clone(),
        )?;
    }

    let mut new_block = TensorBlock::new(
        new_data,
        merged_samples,
        new_components,
        new_properties
    ).expect("invalid block");

    // now collect & merge the different gradients
    for (parameter, first_gradient) in first_block.gradients() {
        let new_gradient_samples = merge_gradient_samples(
            blocks_to_merge, parameter, &samples_mapping
        );

        let mut new_shape = first_gradient.data.shape()?.to_vec();
        new_shape[0] = new_gradient_samples.count();
        let mut new_gradient = first_block.values.data.create(&new_shape)?;
        let new_components = first_gradient.components().to_vec();

        for ((_, block), samples_mapping) in blocks_to_merge.iter().zip(&samples_mapping) {
            let gradient = block.get_gradient(parameter).expect("missing gradient");
            debug_assert!(gradient.components() == new_components);

            let mut samples_to_move = Vec::new();
            for (sample_i, grad_sample) in gradient.samples().iter().enumerate() {
                // translate from the old sample id in gradients to the new ones
                let mut grad_sample = grad_sample.to_vec();
                let old_sample_i = grad_sample[0].usize();
                grad_sample[0] = LabelValue::from(samples_mapping[old_sample_i]);

                let new_sample_i = new_gradient_samples.position(&grad_sample).expect("missing entry in merged samples");
                samples_to_move.push(eqs_sample_move_t {
                    input: sample_i,
                    output: new_sample_i,
                });
            }
            new_gradient.move_samples_from(
                &gradient.data,
                &samples_to_move,
                property_range.clone(),
            )?;
        }

        new_block.add_gradient(
            parameter, new_gradient, new_gradient_samples, new_components
        ).expect("created invalid gradients");
    }

    return Ok(new_block);
}


#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use super::super::utils::example_tensor;

    use ndarray::ArrayD;

    #[test]
    fn sorted_samples() {
        let mut tensor = example_tensor();
        tensor.keys_to_samples(&["key_2"], true).unwrap();

        assert_eq!(tensor.keys().count(), 3);
        assert_eq!(tensor.keys().names(), ["key_1"]);
        assert_eq!(tensor.keys()[0], [LabelValue::new(0)]);
        assert_eq!(tensor.keys()[1], [LabelValue::new(1)]);
        assert_eq!(tensor.keys()[2], [LabelValue::new(2)]);

        assert_eq!(tensor.blocks().len(), 3);

        // The first two blocks are not modified
        let block_1 = &tensor.blocks()[0];
        assert_eq!(block_1.values.data.as_array(), ArrayD::from_elem(vec![3, 1, 1], 1.0));
        assert_eq!(block_1.values.samples().count(), 3);
        assert_eq!(block_1.values.samples()[0], [LabelValue::new(0), LabelValue::new(0)]);
        assert_eq!(block_1.values.samples()[1], [LabelValue::new(2), LabelValue::new(0)]);
        assert_eq!(block_1.values.samples()[2], [LabelValue::new(4), LabelValue::new(0)]);

        let block_2 = &tensor.blocks()[1];
        assert_eq!(block_2.values.data.as_array(), ArrayD::from_elem(vec![3, 1, 3], 2.0));
        assert_eq!(block_2.values.samples().count(), 3);
        assert_eq!(block_2.values.samples()[0], [LabelValue::new(0), LabelValue::new(0)]);
        assert_eq!(block_2.values.samples()[1], [LabelValue::new(1), LabelValue::new(0)]);
        assert_eq!(block_2.values.samples()[2], [LabelValue::new(3), LabelValue::new(0)]);

        // The new third block contains the old third and fourth blocks merged
        let block_3 = &tensor.blocks()[2];
        assert_eq!(block_3.values.samples().names(), ["samples", "key_2"]);
        assert_eq!(block_3.values.samples().count(), 8);
        assert_eq!(block_3.values.samples()[0], [LabelValue::new(0), LabelValue::new(2)]);
        assert_eq!(block_3.values.samples()[1], [LabelValue::new(0), LabelValue::new(3)]);
        assert_eq!(block_3.values.samples()[2], [LabelValue::new(1), LabelValue::new(3)]);
        assert_eq!(block_3.values.samples()[3], [LabelValue::new(2), LabelValue::new(3)]);
        assert_eq!(block_3.values.samples()[4], [LabelValue::new(3), LabelValue::new(2)]);
        assert_eq!(block_3.values.samples()[5], [LabelValue::new(5), LabelValue::new(3)]);
        assert_eq!(block_3.values.samples()[6], [LabelValue::new(6), LabelValue::new(2)]);
        assert_eq!(block_3.values.samples()[7], [LabelValue::new(8), LabelValue::new(2)]);

        assert_eq!(block_3.values.components().len(), 1);
        assert_eq!(block_3.values.components()[0].names(), ["components"]);
        assert_eq!(block_3.values.components()[0].count(), 3);
        assert_eq!(block_3.values.components()[0][0], [LabelValue::new(0)]);
        assert_eq!(block_3.values.components()[0][1], [LabelValue::new(1)]);
        assert_eq!(block_3.values.components()[0][2], [LabelValue::new(2)]);

        assert_eq!(block_3.values.properties().names(), ["properties"]);
        assert_eq!(block_3.values.properties().count(), 1);
        assert_eq!(block_3.values.properties()[0], [LabelValue::new(0)]);

        let expected = ArrayD::from_shape_vec(vec![8, 3, 1], vec![
            3.0, 3.0, 3.0,
            4.0, 4.0, 4.0,
            4.0, 4.0, 4.0,
            4.0, 4.0, 4.0,
            3.0, 3.0, 3.0,
            4.0, 4.0, 4.0,
            3.0, 3.0, 3.0,
            3.0, 3.0, 3.0,
        ]).unwrap();
        assert_eq!(block_3.values.data.as_array(), expected);

        let gradient_3 = block_3.get_gradient("parameter").unwrap();
        assert_eq!(gradient_3.samples().names(), ["sample", "parameter"]);
        assert_eq!(gradient_3.samples().count(), 3);
        assert_eq!(gradient_3.samples()[0], [LabelValue::new(1), LabelValue::new(1)]);
        assert_eq!(gradient_3.samples()[1], [LabelValue::new(4), LabelValue::new(-2)]);
        assert_eq!(gradient_3.samples()[2], [LabelValue::new(5), LabelValue::new(3)]);

        let expected = ArrayD::from_shape_vec(vec![3, 3, 1], vec![
            14.0, 14.0, 14.0,
            13.0, 13.0, 13.0,
            14.0, 14.0, 14.0,
        ]).unwrap();
        assert_eq!(gradient_3.data.as_array(), expected);
    }

    #[test]
    fn unsorted_samples() {
        let mut tensor = example_tensor();
        tensor.keys_to_samples(&["key_2"], false).unwrap();

        let block_3 = &tensor.blocks()[2];
        assert_eq!(block_3.values.samples().names(), ["samples", "key_2"]);
        assert_eq!(block_3.values.samples().count(), 8);
        assert_eq!(block_3.values.samples()[0], [LabelValue::new(0), LabelValue::new(2)]);
        assert_eq!(block_3.values.samples()[1], [LabelValue::new(3), LabelValue::new(2)]);
        assert_eq!(block_3.values.samples()[2], [LabelValue::new(6), LabelValue::new(2)]);
        assert_eq!(block_3.values.samples()[3], [LabelValue::new(8), LabelValue::new(2)]);
        assert_eq!(block_3.values.samples()[4], [LabelValue::new(0), LabelValue::new(3)]);
        assert_eq!(block_3.values.samples()[5], [LabelValue::new(1), LabelValue::new(3)]);
        assert_eq!(block_3.values.samples()[6], [LabelValue::new(2), LabelValue::new(3)]);
        assert_eq!(block_3.values.samples()[7], [LabelValue::new(5), LabelValue::new(3)]);
    }
}
