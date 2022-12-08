use std::sync::Arc;

use indexmap::IndexSet;

use crate::labels::{Labels, LabelsBuilder};
use crate::{Error, TensorBlock};

use crate::data::eqs_sample_mapping_t;

use super::TensorMap;
use super::utils::{KeyAndBlock, remove_variables_from_keys, merge_samples, merge_gradient_samples};


impl TensorMap {
    /// Merge blocks with the same value for selected keys variables along the
    /// property axis.
    ///
    /// The variables (names) of `keys_to_move` will be moved from the keys to
    /// the property labels, and blocks with the same remaining keys variables
    /// will be merged together along the property axis.
    ///
    /// If `keys_to_move` does not contains any entries (`keys_to_move.count()
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
    pub fn keys_to_properties(&self, keys_to_move: &Labels, sort_samples: bool) -> Result<TensorMap, Error> {
        let names_to_move = keys_to_move.names();
        let splitted_keys = remove_variables_from_keys(&self.keys, &names_to_move)?;

        let keys_to_move = if keys_to_move.count() == 0 {
            None
        } else {
            Some(keys_to_move)
        };

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

            let block = merge_blocks_along_properties(
                &blocks_to_merge,
                keys_to_move,
                &names_to_move,
                sort_samples,
            )?;
            new_blocks.push(block);
        } else {
            for entry in splitted_keys.new_keys.iter() {
                let mut selection = LabelsBuilder::new(splitted_keys.new_keys.names());
                selection.add(entry)?;

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

                let block = merge_blocks_along_properties(
                    &blocks_to_merge,
                    keys_to_move,
                    &names_to_move,
                    sort_samples,
                )?;
                new_blocks.push(block);
            }
        }

        return TensorMap::new(splitted_keys.new_keys, new_blocks);
    }
}

/// Merge the given `blocks` along the property axis.
#[allow(clippy::too_many_lines)]
fn merge_blocks_along_properties(
    blocks_to_merge: &[KeyAndBlock],
    keys_to_move: Option<&Labels>,
    extracted_names: &[&str],
    sort_samples: bool,
) -> Result<TensorBlock, Error> {
    assert!(!blocks_to_merge.is_empty());

    let first_block = blocks_to_merge[0].1;
    let first_components_label = &first_block.values().components;
    let first_property_labels = &first_block.values().properties;
    for (_, block) in blocks_to_merge {
        if &block.values().components != first_components_label {
            return Err(Error::InvalidParameter(
                "can not move keys to properties if the blocks have \
                different components labels, call components_to_properties first".into()
            ));
        }

        if keys_to_move.is_some() && &block.values().properties != first_property_labels {
            // TODO: this might be possible but also pretty slow. It would
            // requires to lookup the position of properties one by one in the
            // merged properties
            return Err(Error::InvalidParameter(
                "can not provide values for the keys to move to properties if \
                the blocks have different property labels".into()
            ));
        }
    }

    // collect and merge samples across the blocks
    let (merged_samples, samples_mappings) = merge_samples(
        blocks_to_merge,
        first_block.values().samples.names(),
        sort_samples,
    );

    let mut new_properties = IndexSet::new();
    if let Some(keys_to_move) = keys_to_move {
        // use the user-provided new values
        for new_property in keys_to_move {
            for (_, block) in blocks_to_merge {
                for old_property in block.values().properties.iter() {
                    let mut property = new_property.to_vec();
                    property.extend_from_slice(old_property);
                    new_properties.insert(property);
                }
            }
        }
        assert_eq!(new_properties.len(), first_property_labels.count() * keys_to_move.count());
    } else {
        // collect properties from the blocks, augmenting them with the new
        // properties
        for (new_property, block) in blocks_to_merge {
            for old_property in block.values().properties.iter() {
                let mut property = new_property.clone();
                property.extend_from_slice(old_property);
                new_properties.insert(property);
            }
        }
    }

    let new_property_names = extracted_names.iter()
        .chain(first_block.values().properties.names().iter())
        .copied()
        .collect();
    let mut new_properties_builder = LabelsBuilder::new(new_property_names);
    for property in new_properties {
        new_properties_builder.add(&property)?;
    }

    let new_components = first_block.values().components.to_vec();
    let new_properties = Arc::new(new_properties_builder.finish());
    let new_properties_count = new_properties.count();

    // create a new array and move the data around
    let mut new_shape = first_block.values().data.shape()?.to_vec();
    new_shape[0] = merged_samples.count();
    let property_axis = new_shape.len() - 1;
    new_shape[property_axis] = new_properties_count;
    let mut new_data = first_block.values().data.create(&new_shape)?;

    // compute the property range for each block, i.e. where we want to put
    // the corresponding data
    let mut property_ranges = Vec::new();
    for (new_property, block) in blocks_to_merge {
        if block.values().properties.is_empty() {
            // no properties, ignore this block
            property_ranges.push(None);
            continue;
        }

        let mut first = new_property.clone();
        first.extend_from_slice(&block.values().properties[0]);

        // we can lookup only the `first` new property here, since the new
        // properties match exactly the old ones, just with an added "channel"
        // of the moved key.

        // start can be None is the user requested a set of values which do not
        // include the key for the current block
        let start = new_properties.position(&first);
        let size = block.values().properties.count();
        if let Some(start) = start {
            property_ranges.push(Some(start..(start + size)));
        } else {
            property_ranges.push(None);
        }
    }

    debug_assert_eq!(blocks_to_merge.len(), samples_mappings.len());
    debug_assert_eq!(blocks_to_merge.len(), property_ranges.len());
    // for each block, gather the data to be moved & send it in one go
    for (((_, block), samples_mapping), property_range) in blocks_to_merge.iter().zip(&samples_mappings).zip(&property_ranges) {
        if let Some(property_range) = property_range {
            new_data.move_samples_from(
                &block.values().data,
                samples_mapping,
                property_range.clone()
            )?;
        }
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
        )?;

        let mut new_shape = first_gradient.data.shape()?.to_vec();
        new_shape[0] = new_gradient_samples.count();
        let property_axis = new_shape.len() - 1;
        new_shape[property_axis] = new_properties_count;

        let mut new_gradient = first_block.values().data.create(&new_shape)?;
        let new_components = first_gradient.components.to_vec();

        for (((_, block), samples_mapping), property_range) in blocks_to_merge.iter().zip(&samples_mappings).zip(&property_ranges) {
            if property_range.is_none() {
                continue;
            }
            let property_range = property_range.as_ref().unwrap();

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
