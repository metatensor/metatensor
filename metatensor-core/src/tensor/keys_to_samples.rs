use std::sync::Arc;

use crate::labels::Labels;
use crate::{Error, TensorBlock};

use crate::data::mts_data_movement_t;

use super::TensorMap;
use super::utils::{KeyAndBlock, remove_dimensions_from_keys, merge_samples, merge_gradient_samples};

impl TensorMap {
    /// Merge blocks with the same value for selected keys dimensions along the
    /// samples axis.
    ///
    /// The dimensions (names) of `keys_to_move` will be moved from the keys to
    /// the sample labels, and blocks with the same remaining keys dimensions
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
    /// If the blocks have different property labels, the resulting block will
    /// have the union of all property labels, and values will be padded with
    /// zeros where appropriate.
    pub fn keys_to_samples(&self, keys_to_move: &Labels, sort_samples: bool) -> Result<TensorMap, Error> {
        if self.keys.is_empty() {
            return Err(Error::InvalidParameter(
                "there are no keys to move in an empty TensorMap".into()
            ));
        }

        if keys_to_move.count() > 0 {
            return Err(Error::InvalidParameter(
                "user provided values for the keys to move is not yet implemented, \
                `keys_to_move` should not contain any entry when calling keys_to_samples".into()
            ))
        }

        let names_to_move = keys_to_move.names();
        let splitted_keys = remove_dimensions_from_keys(&self.keys, &names_to_move)?;

        let mut new_blocks = Vec::new();
        if splitted_keys.new_keys.count() == 1 {
            // create a single block with everything
            let blocks_to_merge = self.keys.iter()
                .zip(&self.blocks)
                .map(|(key, block)| {
                    let mut moved_key = Vec::new();
                    for &i in &splitted_keys.dimensions_positions {
                        moved_key.push(key[i]);
                    }

                    KeyAndBlock {
                        key: moved_key,
                        block
                    }
                })
                .collect::<Vec<_>>();

            let block = merge_blocks_along_samples(
                &blocks_to_merge,
                &names_to_move,
                sort_samples,
            )?;
            new_blocks.push(block);
        } else {
            for entry in &splitted_keys.new_keys {
                let selection = Labels::new(
                    &splitted_keys.new_keys.names(),
                    entry.to_vec()
                ).expect("invalid labels");

                let matching = self.blocks_matching(&selection)?;
                let blocks_to_merge = matching.iter()
                    .map(|&i| {
                        let block = &self.blocks[i];
                        let key = &self.keys[i];
                        let mut moved_key = Vec::new();
                        for &i in &splitted_keys.dimensions_positions {
                            moved_key.push(key[i]);
                        }

                        KeyAndBlock {
                            key: moved_key,
                            block
                        }
                    })
                    .collect::<Vec<_>>();

                new_blocks.push(merge_blocks_along_samples(
                    &blocks_to_merge,
                    &names_to_move,
                    sort_samples,
                )?);
            }
        }
        let mut tensor = TensorMap::new(Arc::new(splitted_keys.new_keys), new_blocks)?;
        for (k, v) in &self.info {
            tensor.add_info(k, v.clone());
        }
        return Ok(tensor);
    }
}

/// Merge the given `blocks` along the sample axis.
fn merge_blocks_along_samples(
    blocks_to_merge: &[KeyAndBlock],
    extracted_names: &[&str],
    sort_samples: bool,
) -> Result<TensorBlock, Error> {
    assert!(!blocks_to_merge.is_empty());

    let first_block = blocks_to_merge[0].block;
    for gradient in first_block.gradients().values() {
        if !gradient.gradients().is_empty() {
            return Err(Error::InvalidParameter(
                "gradient of gradients are not supported yet in keys_to_samples".into()
            ));
        }
    }

    let first_components_label = &first_block.components;

    for KeyAndBlock{block, ..} in blocks_to_merge {
        if &block.components != first_components_label {
            return Err(Error::InvalidParameter(
                "can not move keys to samples if the blocks have \
                different components labels, call components_to_properties first".into()
            ))
        }
    }

    // collect and merge samples across the blocks
    let new_sample_names = first_block.samples.names().iter()
        .chain(extracted_names.iter())
        .copied()
        .collect::<Vec<_>>();
    let (merged_samples, samples_mappings) = merge_samples(
        blocks_to_merge,
        &new_sample_names,
        sort_samples,
    );

    let new_components = first_block.components.to_vec();

    // merge properties across the blocks
    let (new_properties, property_mappings) = merge_properties(blocks_to_merge)?;

    debug_assert_eq!(blocks_to_merge.len(), samples_mappings.len());
    debug_assert_eq!(blocks_to_merge.len(), property_mappings.len());

    let new_data = merge_data(
        &first_block.values,
        merged_samples.count(),
        new_properties.count(),
        &property_mappings,
        &samples_mappings,
        &blocks_to_merge.iter().map(|b| &b.block.values).collect::<Vec<_>>(),
    )?;

    let mut new_block = TensorBlock::new(
        new_data,
        merged_samples,
        new_components,
        new_properties.clone()
    ).expect("invalid block");

    // now collect & merge the different gradients
    for (parameter, first_gradient) in first_block.gradients() {
        let new_gradient_samples = merge_gradient_samples(
            blocks_to_merge, parameter, &samples_mappings
        );

        let new_components = first_gradient.components.to_vec();

        let mut gradient_samples_mappings = Vec::new();
        let mut gradients_to_merge = Vec::new();

        for (KeyAndBlock{block, ..}, samples_mapping) in blocks_to_merge.iter().zip(&samples_mappings) {
            let gradient = block.gradient(parameter).expect("missing gradient");
            debug_assert!(*gradient.components == *new_components);

            gradients_to_merge.push(&gradient.values);

            // Gradients share the same properties as the values, so we reuse property_mappings[i]
            // which is handled inside merge_data
            let mut gradient_mapping = Vec::new();
            for grad_sample in gradient.samples.iter() {
                // translate from the old sample id in gradients to the new ones
                let mut grad_sample = grad_sample.to_vec();
                let old_sample_i = usize::try_from(grad_sample[0]).expect("could not convert to usize");

                let new_sample_i = samples_mapping[old_sample_i];
                grad_sample[0] = i32::try_from(new_sample_i).expect("could not convert to i32");

                let new_grad_sample_i = new_gradient_samples.position(&grad_sample).expect("missing entry in merged samples");

                gradient_mapping.push(new_grad_sample_i);
            }
            gradient_samples_mappings.push(gradient_mapping);
        }

        let new_gradient = merge_data(
            &first_gradient.values,
            new_gradient_samples.count(),
            new_properties.count(),
            &property_mappings,
            &gradient_samples_mappings,
            &gradients_to_merge,
        )?;

        let new_gradient = TensorBlock::new(
            new_gradient,
            new_gradient_samples,
            new_components,
            new_block.properties.clone()
        ).expect("created invalid gradient");

        new_block.add_gradient(parameter, new_gradient).expect("could not add gradient");
    }

    return Ok(new_block);
}

fn merge_properties(
    blocks_to_merge: &[KeyAndBlock],
) -> Result<(Arc<Labels>, Vec<Vec<usize>>), Error> {
    let first_block = blocks_to_merge[0].block;
    let mut new_properties = Arc::clone(&first_block.properties);
    let mut property_mappings = Vec::with_capacity(blocks_to_merge.len());

    // mapping for the first block (identity, since new_properties starts as first_block.properties)
    let first_mapping = (0..new_properties.count()).collect::<Vec<_>>();
    property_mappings.push(first_mapping);

    for KeyAndBlock{block, ..} in &blocks_to_merge[1..] {
        let mut mapping_for_current_union = vec![0; new_properties.count()];
        let mut mapping_for_new_block = vec![0; block.properties.count()];

        let next_properties = new_properties.union(
            &block.properties,
            &mut mapping_for_current_union,
            &mut mapping_for_new_block
        )?;

        let mapping_for_current_union = mapping_for_current_union.into_iter()
            .map(|i| usize::try_from(i).expect("mapping contains negative value") )
            .collect::<Vec<_>>();
        let mapping_for_new_block = mapping_for_new_block.into_iter()
            .map(|i| usize::try_from(i).expect("mapping contains negative value") )
            .collect::<Vec<_>>();

        // update existing mappings to point to the new positions in next_properties
        for mapping in &mut property_mappings {
            for entry in mapping.iter_mut() {
                *entry = mapping_for_current_union[*entry];
            }
        }

        // add mapping for the new block
        property_mappings.push(mapping_for_new_block);

        new_properties = Arc::new(next_properties);
    }

    Ok((new_properties, property_mappings))
}

fn merge_data(
    prototype_array: &crate::data::mts_array_t,
    samples_count: usize,
    properties_count: usize,
    property_mappings: &[Vec<usize>],
    samples_mappings: &[Vec<usize>],
    blocks: &[&crate::data::mts_array_t],
) -> Result<crate::data::mts_array_t, Error> {
    let mut new_shape = prototype_array.shape()?.to_vec();
    new_shape[0] = samples_count;
    let len = new_shape.len();
    new_shape[len - 1] = properties_count;

    let mut output = prototype_array.create(&new_shape)?;

    for (i, (block, samples_mapping)) in blocks.iter().zip(samples_mappings).enumerate() {
        let property_mapping = &property_mappings[i];
        let mut movements = Vec::new();

        for (sample_i, &new_sample_i) in samples_mapping.iter().enumerate() {
            // we need to decompose the property mapping into contiguous chunks
            let mut prop_i = 0;
            while prop_i < property_mapping.len() {
                let start_in = prop_i;
                let start_out = property_mapping[prop_i];

                let mut length = 1;
                while prop_i + length < property_mapping.len() {
                    if property_mapping[prop_i + length] == start_out + length {
                        length += 1;
                    } else {
                        break;
                    }
                }

                movements.push(mts_data_movement_t {
                    sample_in: sample_i,
                    sample_out: new_sample_i,
                    properties_start_in: start_in,
                    properties_start_out: start_out,
                    properties_length: length,
                });

                prop_i += length;
            }
        }
        output.move_data(block, &movements)?;
    }
    Ok(output)
}
