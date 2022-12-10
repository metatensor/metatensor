use std::sync::Arc;

use crate::labels::{Labels, LabelsBuilder};
use crate::{Error, TensorBlock};

use crate::data::eqs_sample_mapping_t;

use super::TensorMap;
use super::utils::{KeyAndBlock, remove_variables_from_keys, merge_samples, merge_gradient_samples};

impl TensorMap {
    /// Merge blocks with the same value for selected keys variables along the
    /// samples axis.
    ///
    /// The variables (names) of `keys_to_move` will be moved from the keys to
    /// the sample labels, and blocks with the same remaining keys variables
    /// will be merged together along the sample axis.
    ///
    /// `keys_to_move` must be empty (`keys_to_move.count() == 0`), and the new
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
    pub fn keys_to_samples(&self, keys_to_move: &Labels, sort_samples: bool) -> Result<TensorMap, Error> {
        if keys_to_move.count() > 0 {
            return Err(Error::InvalidParameter(
                "user provided values for the keys to move is not yet implemented, \
                `keys_to_move` should not contain any entry when calling keys_to_samples".into()
            ))
        }

        let names_to_move = keys_to_move.names();
        let splitted_keys = remove_variables_from_keys(&self.keys, &names_to_move)?;

        let mut new_blocks = Vec::new();
        if splitted_keys.new_keys.count() == 1 {
            // create a single block with everything
            let blocks_to_merge = self.keys.iter()
                .zip(&self.blocks)
                .map(|(key, block)| {
                    let mut moved_key = Vec::new();
                    for &i in &splitted_keys.variables_positions {
                        moved_key.push(key[i]);
                    }

                    (moved_key, block)
                })
                .collect::<Vec<_>>();

            let block = merge_blocks_along_samples(
                &blocks_to_merge,
                &names_to_move,
                sort_samples,
            )?;
            new_blocks.push(block);
        } else {
            for entry in splitted_keys.new_keys.iter() {
                let mut selection = LabelsBuilder::new(splitted_keys.new_keys.names());
                selection.add(entry);

                let matching = self.blocks_matching(&selection.finish())?;
                let blocks_to_merge = matching.iter()
                    .map(|&i| {
                        let block = &self.blocks[i];
                        let key = &self.keys[i];
                        let mut moved_key = Vec::new();
                        for &i in &splitted_keys.variables_positions {
                            moved_key.push(key[i]);
                        }

                        (moved_key, block)
                    })
                    .collect::<Vec<_>>();

                new_blocks.push(merge_blocks_along_samples(
                    &blocks_to_merge,
                    &names_to_move,
                    sort_samples,
                )?);
            }
        }

        return TensorMap::new(splitted_keys.new_keys, new_blocks);
    }
}

/// Merge the given `blocks` along the sample axis.
fn merge_blocks_along_samples(
    blocks_to_merge: &[KeyAndBlock],
    extracted_names: &[&str],
    sort_samples: bool,
) -> Result<TensorBlock, Error> {
    assert!(!blocks_to_merge.is_empty());

    let first_block = blocks_to_merge[0].1;
    let first_components_label = &first_block.values().components;
    let first_properties_label = &first_block.values().properties;

    for (_, block) in blocks_to_merge {
        if &block.values().components != first_components_label {
            return Err(Error::InvalidParameter(
                "can not move keys to samples if the blocks have \
                different components labels, call components_to_properties first".into()
            ))
        }

        if &block.values().properties != first_properties_label {
            return Err(Error::InvalidParameter(
                "can not move keys to samples if the blocks have \
                different property labels".into() // TODO: this might be possible
            ))
        }
    }

    // collect and merge samples across the blocks
    let new_samples_names = first_block.values().samples.names().iter()
        .chain(extracted_names.iter())
        .copied()
        .collect();
    let (merged_samples, samples_mappings) = merge_samples(
        blocks_to_merge,
        new_samples_names,
        sort_samples,
    );

    let new_components = first_block.values().components.to_vec();
    let new_properties = Arc::clone(&first_block.values().properties);

    let mut new_shape = first_block.values().data.shape()?.to_vec();
    new_shape[0] = merged_samples.count();
    let mut new_data = first_block.values().data.create(&new_shape)?;

    let property_range = 0..new_properties.count();

    debug_assert_eq!(blocks_to_merge.len(), samples_mappings.len());
    for ((_, block), samples_mapping) in blocks_to_merge.iter().zip(&samples_mappings) {
        new_data.move_samples_from(
            &block.values().data,
            samples_mapping,
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
            blocks_to_merge, parameter, &samples_mappings
        );

        let mut new_shape = first_gradient.data.shape()?.to_vec();
        new_shape[0] = new_gradient_samples.count();
        let mut new_gradient = first_block.values().data.create(&new_shape)?;
        let new_components = first_gradient.components.to_vec();

        for ((_, block), samples_mapping) in blocks_to_merge.iter().zip(&samples_mappings) {
            let gradient = block.gradient(parameter).expect("missing gradient");
            debug_assert!(*gradient.components == *new_components);

            let mut samples_to_move = Vec::new();
            for (sample_i, grad_sample) in gradient.samples.iter().enumerate() {
                // translate from the old sample id in gradients to the new ones
                let mut grad_sample = grad_sample.to_vec();
                let old_sample_i = grad_sample[0].usize();

                let mapping = &samples_mapping[old_sample_i];
                debug_assert_eq!(mapping.input, old_sample_i);
                grad_sample[0] = mapping.output.into();

                let new_sample_i = new_gradient_samples.position(&grad_sample).expect("missing entry in merged samples");
                samples_to_move.push(eqs_sample_mapping_t {
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
    use crate::{LabelsBuilder, Labels, TensorMap};
    use super::super::utils::{example_tensor, example_block};

    use ndarray::ArrayD;

    #[test]
    fn sorted_samples() {
        let keys_to_move = LabelsBuilder::new(vec!["key_2"]).finish();
        let tensor = example_tensor().keys_to_samples(&keys_to_move, true).unwrap();

        assert_eq!(tensor.keys().count(), 3);
        assert_eq!(tensor.keys().names(), ["key_1"]);
        assert_eq!(tensor.keys()[0], [0]);
        assert_eq!(tensor.keys()[1], [1]);
        assert_eq!(tensor.keys()[2], [2]);

        assert_eq!(tensor.blocks().len(), 3);

        // The first two blocks are not modified
        let block_1 = &tensor.blocks()[0];
        assert_eq!(block_1.values().data.as_array(), ArrayD::from_elem(vec![3, 1, 1], 1.0));
        assert_eq!(block_1.values().samples.count(), 3);
        assert_eq!(block_1.values().samples[0], [0, 0]);
        assert_eq!(block_1.values().samples[1], [2, 0]);
        assert_eq!(block_1.values().samples[2], [4, 0]);

        let block_2 = &tensor.blocks()[1];
        assert_eq!(block_2.values().data.as_array(), ArrayD::from_elem(vec![3, 1, 3], 2.0));
        assert_eq!(block_2.values().samples.count(), 3);
        assert_eq!(block_2.values().samples[0], [0, 0]);
        assert_eq!(block_2.values().samples[1], [1, 0]);
        assert_eq!(block_2.values().samples[2], [3, 0]);

        // The new third block contains the old third and fourth blocks merged
        let block_3 = &tensor.blocks()[2];
        assert_eq!(block_3.values().samples.names(), ["samples", "key_2"]);
        assert_eq!(block_3.values().samples.count(), 8);
        assert_eq!(block_3.values().samples[0], [0, 2]);
        assert_eq!(block_3.values().samples[1], [0, 3]);
        assert_eq!(block_3.values().samples[2], [1, 3]);
        assert_eq!(block_3.values().samples[3], [2, 3]);
        assert_eq!(block_3.values().samples[4], [3, 2]);
        assert_eq!(block_3.values().samples[5], [5, 3]);
        assert_eq!(block_3.values().samples[6], [6, 2]);
        assert_eq!(block_3.values().samples[7], [8, 2]);

        assert_eq!(block_3.values().components.len(), 1);
        assert_eq!(block_3.values().components[0].names(), ["components"]);
        assert_eq!(block_3.values().components[0].count(), 3);
        assert_eq!(block_3.values().components[0][0], [0]);
        assert_eq!(block_3.values().components[0][1], [1]);
        assert_eq!(block_3.values().components[0][2], [2]);

        assert_eq!(block_3.values().properties.names(), ["properties"]);
        assert_eq!(block_3.values().properties.count(), 1);
        assert_eq!(block_3.values().properties[0], [0]);

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
        assert_eq!(block_3.values().data.as_array(), expected);

        let gradient_3 = block_3.gradient("parameter").unwrap();
        assert_eq!(gradient_3.samples.names(), ["sample", "parameter"]);
        assert_eq!(gradient_3.samples.count(), 3);
        assert_eq!(gradient_3.samples[0], [1, 1]);
        assert_eq!(gradient_3.samples[1], [4, -2]);
        assert_eq!(gradient_3.samples[2], [5, 3]);

        let expected = ArrayD::from_shape_vec(vec![3, 3, 1], vec![
            14.0, 14.0, 14.0,
            13.0, 13.0, 13.0,
            14.0, 14.0, 14.0,
        ]).unwrap();
        assert_eq!(gradient_3.data.as_array(), expected);
    }

    #[test]
    fn unsorted_samples() {
        let keys_to_move = LabelsBuilder::new(vec!["key_2"]).finish();
        let tensor = example_tensor().keys_to_samples(&keys_to_move, false).unwrap();

        let block_3 = &tensor.blocks()[2];
        assert_eq!(block_3.values().samples.names(), ["samples", "key_2"]);
        assert_eq!(block_3.values().samples.count(), 8);
        assert_eq!(block_3.values().samples[0], [0, 2]);
        assert_eq!(block_3.values().samples[1], [3, 2]);
        assert_eq!(block_3.values().samples[2], [6, 2]);
        assert_eq!(block_3.values().samples[3], [8, 2]);
        assert_eq!(block_3.values().samples[4], [0, 3]);
        assert_eq!(block_3.values().samples[5], [1, 3]);
        assert_eq!(block_3.values().samples[6], [2, 3]);
        assert_eq!(block_3.values().samples[7], [5, 3]);
    }

    #[test]
    fn user_provided_entries() {
        let mut keys_to_move = LabelsBuilder::new(vec!["key_2"]);
        keys_to_move.add(&[3]);
        let result = example_tensor().keys_to_samples(&keys_to_move.finish(), false);

        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: user provided values for the keys to move is \
            not yet implemented, `keys_to_move` should not contain any entry \
            when calling keys_to_samples"
        );
    }

    #[test]
    #[allow(clippy::vec_init_then_push)]
    fn empty_samples() {
        let mut blocks = Vec::new();
        blocks.push(example_block(
            /* samples          */ vec![[0], [2], [4]],
            /* components       */ vec![[0]],
            /* properties       */ vec![[0], [1], [2], [3]],
            /* gradient_samples */ vec![],
            /* values           */ 1.0,
            /* gradient_values  */ 11.0,
        ));

        blocks.push(example_block(
            /* samples          */ vec![],
            /* components       */ vec![[0]],
            /* properties       */ vec![[0], [1], [2], [3]],
            /* gradient_samples */ vec![],
            /* values           */ 0.0,
            /* gradient_values  */ 0.0,
        ));
        let mut keys = LabelsBuilder::new(vec!["key_1", "key_2"]);
        keys.add(&[0, 0]);
        keys.add(&[1, 0]);
        let keys = keys.finish();

        let tensor = TensorMap::new(keys, blocks).unwrap();

        let keys_to_move = Labels::empty(vec!["key_1"]);
        let moved = tensor.keys_to_samples(&keys_to_move, true).unwrap();

        assert_eq!(
            *moved.block_by_id(0).values().samples,
            Labels::new(["samples", "key_1"], &[
                [0, 0],
                [2, 0],
                [4, 0],
            ])
        );
    }
}
