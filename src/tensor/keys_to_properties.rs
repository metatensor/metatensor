use std::sync::Arc;

use indexmap::IndexSet;

use crate::labels::{Labels, LabelsBuilder};
use crate::{Error, TensorBlock};

use crate::data::eqs_sample_mapping_t;

use super::TensorMap;
use super::utils::{KeyAndBlock, split_keys, merge_samples, merge_gradient_samples};


impl TensorMap {
    /// Move the given variables from the keys to the property labels of the
    /// blocks.
    ///
    /// Blocks containing the same values in the keys for the `variables` will
    /// be merged together. The resulting merged blocks will have `variables` as
    /// the first property variables, followed by the current properties. The
    /// new sample labels will contains all of the merged blocks sample labels.
    ///
    /// The order of the samples is controlled by `sort_samples`. If
    /// `sort_samples` is true, samples are re-ordered to keep them
    /// lexicographically sorted. Otherwise they are kept in the order in which
    /// they appear in the blocks.
    pub fn keys_to_properties(&mut self, variables: &[&str], sort_samples: bool) -> Result<(), Error> {
        // TODO: requested values
        // TODO: keys_to_properties_no_gradients?

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

            let block = merge_blocks_along_properties(
                &blocks_to_merge,
                &splitted_keys.extracted_keys,
                sort_samples,
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

                let block = merge_blocks_along_properties(
                    &blocks_to_merge,
                    &splitted_keys.extracted_keys,
                    sort_samples,
                )?;
                new_blocks.push(block);
            }
        }


        self.keys = splitted_keys.new_keys;
        self.blocks = new_blocks;

        return Ok(());
    }
}

/// Merge the given `blocks` along the property axis.
#[allow(clippy::too_many_lines)]
fn merge_blocks_along_properties(
    blocks_to_merge: &[KeyAndBlock],
    extracted_properties: &Labels,
    sort_samples: bool,
) -> Result<TensorBlock, Error> {
    assert!(!blocks_to_merge.is_empty());

    let first_block = blocks_to_merge[0].1;
    let first_components_label = first_block.values.components();
    for (_, block) in blocks_to_merge {
        if block.values.components() != first_components_label {
            return Err(Error::InvalidParameter(
                "can not move keys to properties if the blocks have \
                different components labels, call components_to_properties first".into()
            ))
        }
    }

    // collect and merge samples across the blocks
    let (merged_samples, samples_mappings) = merge_samples(
        blocks_to_merge,
        first_block.values.samples.names(),
        sort_samples,
    );

    dbg!(&samples_mappings);

    // collect properties across the blocks, augmenting them with the new
    // properties
    let mut new_properties = IndexSet::new();
    for (new_property, block) in blocks_to_merge {
        for old_property in block.values.properties().iter() {
            let mut property = new_property.clone();
            property.extend_from_slice(old_property);
            new_properties.insert(property);
        }
    }

    let new_property_names = extracted_properties.names().iter()
        .chain(first_block.values.properties().names().iter())
        .copied()
        .collect();
    let mut new_properties_builder = LabelsBuilder::new(new_property_names);
    for property in new_properties {
        new_properties_builder.add(property);
    }

    let new_components = first_block.values.components().to_vec();
    let new_properties = Arc::new(new_properties_builder.finish());
    let new_properties_count = new_properties.count();

    // create a new array and move the data around
    let mut new_shape = first_block.values.data.shape()?.to_vec();
    new_shape[0] = merged_samples.count();
    let property_axis = new_shape.len() - 1;
    new_shape[property_axis] = new_properties_count;
    let mut new_data = first_block.values.data.create(&new_shape)?;

    // compute the property range for each block, i.e. where we want to put
    // the corresponding data
    let mut property_ranges = Vec::new();
    for (new_property, block) in blocks_to_merge {
        let size = block.values.properties().count();

        let mut first = new_property.clone();
        first.extend_from_slice(&block.values.properties()[0]);

        let start = new_properties.position(&first).expect("missing the first merged property");

        property_ranges.push(start..(start + size));
    }

    debug_assert_eq!(blocks_to_merge.len(), samples_mappings.len());
    debug_assert_eq!(blocks_to_merge.len(), property_ranges.len());
    // for each block, gather the data to be moved & send it in one go
    for (((_, block), samples_mapping), property_range) in blocks_to_merge.iter().zip(&samples_mappings).zip(&property_ranges) {
        new_data.move_samples_from(
            &block.values.data,
            samples_mapping,
            property_range.clone()
        )?;
    }

    let mut new_block = TensorBlock::new(
        new_data,
        merged_samples,
        new_components,
        new_properties
    ).expect("constructed an invalid block");

    // now collect & merge the different gradients
    for (parameter, first_gradient) in first_block.gradients() {
        let new_gradient_samples = merge_gradient_samples(
            blocks_to_merge, parameter, &samples_mappings
        );

        let mut new_shape = first_gradient.data.shape()?.to_vec();
        new_shape[0] = new_gradient_samples.count();
        let property_axis = new_shape.len() - 1;
        new_shape[property_axis] = new_properties_count;

        let mut new_gradient = first_block.values.data.create(&new_shape)?;
        let new_components = first_gradient.components().to_vec();

        for (((_, block), samples_mapping), property_range) in blocks_to_merge.iter().zip(&samples_mappings).zip(&property_ranges) {
            let gradient = block.get_gradient(parameter).expect("missing gradient");
            debug_assert!(gradient.components() == new_components);

            let mut samples_to_move = Vec::new();
            for (sample_i, grad_sample) in gradient.samples().iter().enumerate() {
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
    use crate::labels::LabelValue;
    use super::super::utils::example_tensor;

    use ndarray::ArrayD;

    #[test]
    fn sorted_samples() {
        let mut tensor = example_tensor();
        tensor.keys_to_properties(&["key_1"], true).unwrap();

        assert_eq!(tensor.keys().count(), 3);
        assert_eq!(tensor.keys().names(), ["key_2"]);
        assert_eq!(tensor.keys()[0], [LabelValue::new(0)]);
        assert_eq!(tensor.keys()[1], [LabelValue::new(2)]);
        assert_eq!(tensor.keys()[2], [LabelValue::new(3)]);

        assert_eq!(tensor.blocks().len(), 3);

        // The new first block contains the old first two blocks merged
        let block_1 = &tensor.blocks()[0];
        assert_eq!(block_1.values.samples().names(), ["samples"]);
        assert_eq!(block_1.values.samples().count(), 5);
        assert_eq!(block_1.values.samples()[0], [LabelValue::new(0)]);
        assert_eq!(block_1.values.samples()[1], [LabelValue::new(1)]);
        assert_eq!(block_1.values.samples()[2], [LabelValue::new(2)]);
        assert_eq!(block_1.values.samples()[3], [LabelValue::new(3)]);
        assert_eq!(block_1.values.samples()[4], [LabelValue::new(4)]);

        assert_eq!(block_1.values.components().len(), 1);
        assert_eq!(block_1.values.components()[0].names(), ["components"]);
        assert_eq!(block_1.values.components()[0].count(), 1);
        assert_eq!(block_1.values.components()[0][0], [LabelValue::new(0)]);

        assert_eq!(block_1.values.properties().names(), ["key_1", "properties"]);
        assert_eq!(block_1.values.properties().count(), 4);
        assert_eq!(block_1.values.properties()[0], [LabelValue::new(0), LabelValue::new(0)]);
        assert_eq!(block_1.values.properties()[1], [LabelValue::new(1), LabelValue::new(3)]);
        assert_eq!(block_1.values.properties()[2], [LabelValue::new(1), LabelValue::new(4)]);
        assert_eq!(block_1.values.properties()[3], [LabelValue::new(1), LabelValue::new(5)]);

        let expected = ArrayD::from_shape_vec(vec![5, 1, 4], vec![
            1.0, 2.0, 2.0, 2.0,
            0.0, 2.0, 2.0, 2.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 2.0, 2.0,
            1.0, 0.0, 0.0, 0.0,
        ]).unwrap();
        assert_eq!(block_1.values.data.as_array(), expected);

        let gradient_1 = block_1.get_gradient("parameter").unwrap();
        assert_eq!(gradient_1.samples().names(), ["sample", "parameter"]);
        assert_eq!(gradient_1.samples().count(), 4);
        assert_eq!(gradient_1.samples()[0], [LabelValue::new(0), LabelValue::new(-2)]);
        assert_eq!(gradient_1.samples()[1], [LabelValue::new(0), LabelValue::new(3)]);
        assert_eq!(gradient_1.samples()[2], [LabelValue::new(3), LabelValue::new(-2)]);
        assert_eq!(gradient_1.samples()[3], [LabelValue::new(4), LabelValue::new(3)]);

        let expected = ArrayD::from_shape_vec(vec![4, 1, 4], vec![
            11.0, 12.0, 12.0, 12.0,
            0.0, 12.0, 12.0, 12.0,
            0.0, 12.0, 12.0, 12.0,
            11.0, 0.0, 0.0, 0.0,
        ]).unwrap();
        assert_eq!(gradient_1.data.as_array(), expected);

        // The new second block contains the old third block
        let block_2 = &tensor.blocks()[1];
        assert_eq!(block_2.values.data.shape().unwrap(), [4, 3, 1]);
        assert_eq!(block_2.values.data.as_array(), ArrayD::from_elem(vec![4, 3, 1], 3.0));
        assert_eq!(block_2.values.properties().count(), 1);
        assert_eq!(block_2.values.properties()[0], [LabelValue::new(2), LabelValue::new(0)]);

        // The new third block contains the old second block
        let block_3 = &tensor.blocks()[2];
        assert_eq!(block_3.values.data.as_array(), ArrayD::from_elem(vec![4, 3, 1], 4.0));
        assert_eq!(block_3.values.properties().count(), 1);
        assert_eq!(block_3.values.properties()[0], [LabelValue::new(2), LabelValue::new(0)]);

    }

    #[test]
    fn unsorted_samples() {
        let mut tensor = example_tensor();
        tensor.keys_to_properties(&["key_1"], false).unwrap();

        assert_eq!(tensor.keys().count(), 3);
        assert_eq!(tensor.blocks().len(), 3);

        let block_1 = &tensor.blocks()[0];
        assert_eq!(block_1.values.samples().names(), ["samples"]);
        assert_eq!(block_1.values.samples().count(), 5);
        assert_eq!(block_1.values.samples()[0], [LabelValue::new(0)]);
        assert_eq!(block_1.values.samples()[1], [LabelValue::new(2)]);
        assert_eq!(block_1.values.samples()[2], [LabelValue::new(4)]);
        assert_eq!(block_1.values.samples()[3], [LabelValue::new(1)]);
        assert_eq!(block_1.values.samples()[4], [LabelValue::new(3)]);
    }
}
