use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use indexmap::IndexSet;

use crate::{Block, Error};
use crate::{Labels, LabelsBuilder, LabelValue};

/// A descriptor is the main user-facing struct of this library, and can store
/// any kind of data used in atomistic machine learning.
///
/// A descriptor contains a list of `Block`s, each one associated with a label
/// -- called sparse labels.
///
/// A descriptor provides functions to move some of these sparse labels to the
/// samples or features labels of the blocks, moving from a sparse
/// representation of the data to a dense one.
#[derive(Debug)]
pub struct Descriptor {
    sparse: Labels,
    blocks: Vec<Block>,
    // TODO: arbitrary descriptor level metadata? e.g. using `HashMap<String, String>`
}

impl Descriptor {
    pub fn new(sparse: Labels, blocks: Vec<Block>) -> Result<Descriptor, Error> {
        if blocks.len() != sparse.count() {
            return Err(Error::InvalidParameter(format!(
                "expected the same number of blocks ({}) as the number of \
                entries in the labels when creating a descriptor, got {}",
                sparse.count(), blocks.len()
            )))
        }

        if !blocks.is_empty() {
            // make sure all blocks have the same kind of samples, symmetric &
            // features labels
            let samples_names = blocks[0].values.samples().names();
            let symmetric_names = blocks[0].values.symmetric().names();
            let features_names = blocks[0].values.features().names();

            let gradients_sample_names = blocks[0].gradients_list().iter()
                .map(|&name| {
                    let gradient = blocks[0].get_gradient(name).expect("missing gradient");
                    (name, gradient.samples().names())
                })
                .collect::<HashMap<_, _>>();


            for block in &blocks {
                if block.values.samples().names() != samples_names {
                    return Err(Error::InvalidParameter(format!(
                        "all blocks must have the same samples labels names, got [{}] and [{}]",
                        block.values.samples().names().join(", "),
                        samples_names.join(", "),
                    )));
                }

                if block.values.symmetric().names() != symmetric_names {
                    return Err(Error::InvalidParameter(format!(
                        "all blocks must have the same symmetric labels names, got [{}] and [{}]",
                        block.values.symmetric().names().join(", "),
                        symmetric_names.join(", "),
                    )));
                }

                if block.values.features().names() != features_names {
                    return Err(Error::InvalidParameter(format!(
                        "all blocks must have the same features labels names, got [{}] and [{}]",
                        block.values.features().names().join(", "),
                        features_names.join(", "),
                    )));
                }

                let gradients_list = block.gradients_list();
                if gradients_list.len() != gradients_sample_names.len() {
                    return Err(Error::InvalidParameter(
                        "all blocks must contains the same set of gradients".into(),
                    ));
                }

                for gradient_name in gradients_list {
                    let gradient = block.get_gradient(gradient_name).expect("missing gradient");

                    match gradients_sample_names.get(gradient_name) {
                        None => {
                            return Err(Error::InvalidParameter(format!(
                                "missing gradient with respect to {} in one of the blocks",
                                gradient_name
                            )));
                        },
                        Some(gradients_sample_names) => {
                            if &gradient.samples().names() != gradients_sample_names {
                                return Err(Error::InvalidParameter(format!(
                                    "all blocks must have the same sample labels names, got [{}] and [{}] for gradients with respect to {}",
                                    gradient.samples().names().join(", "),
                                    gradients_sample_names.join(", "),
                                    gradient_name,
                                )));
                            }
                        }
                    }
                }
            }
        }

        Ok(Descriptor {
            sparse,
            blocks,
        })
    }

    /// Get the list of blocks in this `Descriptor`
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    /// Get mutable access to the list of blocks in this `Descriptor`
    pub fn blocks_mut(&mut self) -> &mut [Block] {
        // TODO: this allow the user to add gradients to only a subset of blocks
        &mut self.blocks
    }

    /// Get the sparse labels associated with this descriptor
    pub fn sparse(&self) -> &Labels {
        &self.sparse
    }

    /// Get an iterator over the label + associated block
    pub fn iter(&self) -> impl Iterator<Item=(&[LabelValue], &Block)> + '_ {
        self.sparse.iter().zip(&self.blocks)
    }

    /// Get an iterator over the label + associated block as a mutable reference
    pub fn iter_mut(&mut self) -> impl Iterator<Item=(&[LabelValue], &mut Block)> + '_ {
        self.sparse.iter().zip(&mut self.blocks)
    }

    /// Get the list of blocks matching the given selection. The selection must
    /// contains a single entry, defining the requested values of the sparse
    /// labels. The selection can contain only a subset of the variables defined
    /// in the sparse labels, in which case there can be multiple matching
    /// blocks.
    pub fn blocks_matching(&self, selection: &Labels) -> Result<Vec<&Block>, Error> {
        let matching = self.find_matching_blocks(selection)?;

        return Ok(matching.into_iter().map(|i| &self.blocks[i]).collect());
    }

    /// Get the list of blocks matching the given selection as mutable
    /// references.
    ///
    /// This function behaves similarly to `blocks_matching`, see the
    /// corresponding documentation.
    pub fn blocks_matching_mut<'a>(&'a mut self, selection: &Labels) -> Result<Vec<&'a mut Block>, Error> {
        let matching = self.find_matching_blocks(selection)?;

        // ensure that the matching indexes are unique
        debug_assert!(matching.iter().collect::<BTreeSet<_>>().len() == matching.len());

        let mut result = Vec::new();
        let blocks_ptr = self.blocks.as_mut_ptr();
        for i in matching {
            // SAFETY: we checked above that the indexes in matching are unique,
            // ensuring we only give out exclusive references.
            unsafe {
                result.push(&mut *blocks_ptr.add(i));
            }
        }

        return Ok(result);
    }

    /// Get a reference to the block matching the given selection.
    ///
    /// The selection behaves similarly to `blocks_matching`, with the exception
    /// that this function returns an error if there is more than one matching
    /// block.
    pub fn block(&self, selection: &Labels) -> Result<&Block, Error> {
        let matching = self.find_matching_blocks(selection)?;
        if matching.len() != 1 {
            let selection_str = selection.names()
                .iter().zip(&selection[0])
                .map(|(name, value)| format!("{} = {}", name, value))
                .collect::<Vec<_>>()
                .join(", ");


            return Err(Error::InvalidParameter(format!(
                "{} blocks matched the selection ({}), expected only one",
                matching.len(), selection_str
            )));
        }

        return Ok(&self.blocks[matching[0]]);
    }

    /// Get a mutable reference to the block matching the given selection.
    ///
    /// The selection behaves similarly to `blocks_matching`, with the exception
    /// that this function returns an error if there is more than one matching
    /// block.
    pub fn block_mut(&mut self, selection: &Labels) -> Result<&mut Block, Error> {
        let matching = self.find_matching_blocks(selection)?;
        if matching.len() != 1 {
            let selection_str = selection.names()
                .iter().zip(&selection[0])
                .map(|(name, value)| format!("{} = {}", name, value))
                .collect::<Vec<_>>()
                .join(", ");


            return Err(Error::InvalidParameter(format!(
                "{} blocks matched the selection ({}), expected only one",
                matching.len(), selection_str
            )));
        }

        return Ok(&mut self.blocks[matching[0]]);
    }

    /// Actual implementation of `blocks_matching` and related functions, this
    /// function finds the matching blocks & return their index in the
    /// `self.blocks` vector.
    fn find_matching_blocks(&self, selection: &Labels) -> Result<Vec<usize>, Error> {
        if selection.size() == 0 {
            return Ok((0..self.blocks().len()).collect());
        }

        if selection.count() != 1 {
            return Err(Error::InvalidParameter(format!(
                "block selection labels must contain a single row, got {}",
                selection.count()
            )));
        }

        let mut variables = Vec::new();
        'outer: for requested in selection.names() {
            for (i, &name) in self.sparse.names().iter().enumerate() {
                if requested == name {
                    variables.push(i);
                    continue 'outer;
                }
            }

            return Err(Error::InvalidParameter(format!(
                "'{}' is not part of the sparse labels for this descriptor",
                requested
            )));
        }

        let mut matching = Vec::new();
        let selection = selection.iter().next().expect("empty selection");

        for (block_i, labels) in self.sparse.iter().enumerate() {
            let mut selected = true;
            for (&requested_i, &value) in variables.iter().zip(selection) {
                if labels[requested_i] != value {
                    selected = false;
                    break;
                }
            }

            if selected {
                matching.push(block_i);
            }
        }

        return Ok(matching);
    }

    /// Move the given variables from the sparse labels to the feature labels of
    /// the blocks.
    ///
    /// The current blocks will be merged together according to the sparse
    /// labels remaining after removing `variables`. The resulting merged blocks
    /// will have `variables` as the first feature variables, followed by the
    /// current features. The new sample labels will contains all of the merged
    /// blocks sample labels, re-ordered to keep them lexicographically sorted.
    pub fn sparse_to_features(&mut self, variables: Vec<&str>) -> Result<(), Error> {
        // TODO: requested values
        // TODO: sparse_to_features_no_gradients?

        if variables.is_empty() {
            return Ok(());
        }

        let (new_sparse, new_features) = self.split_sparse_label(variables)?;

        let mut new_blocks = Vec::new();
        if new_sparse.count() == 1 {
            // create a single block with everything
            let mut matching = Vec::new();
            for i in 0..self.blocks.len() {
                matching.push(i);
            }

            let block = self.merge_blocks_along_features(&matching, &new_features)?;
            new_blocks.push(block);
        } else {
            for entry in new_sparse.iter() {
                let mut selection = LabelsBuilder::new(new_sparse.names());
                selection.add(entry.to_vec());

                let matching = self.find_matching_blocks(&selection.finish())?;
                new_blocks.push(self.merge_blocks_along_features(&matching, &new_features)?);
            }
        }


        self.sparse = new_sparse;
        self.blocks = new_blocks;

        return Ok(());
    }

    /// Split the current sparse labels into a new set of sparse label without
    /// the `variables`; and Labels containing the values taken by `variables`
    /// in the current sparse labels
    fn split_sparse_label(&self, variables: Vec<&str>) -> Result<(Labels, Labels), Error> {
        let sparse_names = self.sparse.names();
        for variable in &variables {
            if !sparse_names.contains(variable) {
                return Err(Error::InvalidParameter(format!(
                    "'{}' is not part of the sparse labels for this descriptor",
                    variable
                )));
            }
        }

        // TODO: use Labels instead of Vec<&str> for variables to ensure
        // uniqueness of variables names & pass 'requested' values around

        let mut remaining = Vec::new();
        let mut remaining_i = Vec::new();
        let mut variables_i = Vec::new();

        'outer: for (i, &name) in sparse_names.iter().enumerate() {
            for &variable in &variables {
                if variable == name {
                    variables_i.push(i);
                    continue 'outer;
                }
            }
            remaining.push(name);
            remaining_i.push(i);
        }

        let mut variables_values = IndexSet::new();
        let mut new_sparse = IndexSet::new();
        for entry in self.sparse.iter() {
            let mut label = Vec::new();
            for &i in &variables_i {
                label.push(entry[i]);
            }
            variables_values.insert(label);

            if !remaining_i.is_empty() {
                let mut label = Vec::new();
                for &i in &remaining_i {
                    label.push(entry[i]);
                }
                new_sparse.insert(label);
            }
        }

        let new_sparse = if new_sparse.is_empty() {
            Labels::single()
        } else {
            let mut new_sparse_builder = LabelsBuilder::new(remaining);
            for entry in new_sparse {
                new_sparse_builder.add(entry);
            }
            new_sparse_builder.finish()
        };

        assert!(!variables_values.is_empty());
        let mut variables_builder = LabelsBuilder::new(variables);
        for entry in variables_values {
            variables_builder.add(entry);
        }

        return Ok((new_sparse, variables_builder.finish()));
    }

    /// Merge the blocks with the given `block_idx` along the feature axis. The
    /// new feature names & values to add to the feature axis are passed in
    /// `new_feature_labels`.
    fn merge_blocks_along_features(&self,
        block_idx: &[usize],
        new_feature_labels: &Labels,
    ) -> Result<Block, Error> {
        assert!(!block_idx.is_empty());

        let blocks_to_merge = block_idx.iter().map(|&i| &self.blocks[i]).collect::<Vec<_>>();

        let first_block = &self.blocks[block_idx[0]];
        let first_symmetric_label = first_block.values.symmetric();
        for block in &blocks_to_merge {
            if block.values.symmetric() != first_symmetric_label {
                return Err(Error::InvalidParameter(
                    "can not move sparse label to features if the blocks have \
                    different symmetric labels, call symmetric_to_features first".into()
                ))
            }
        }

        let new_feature_names = new_feature_labels.names().iter()
            .chain(first_block.values.features().names().iter())
            .copied()
            .collect();
        let mut new_features_builder = LabelsBuilder::new(new_feature_names);
        let mut old_feature_sizes = Vec::new();

        // we need to collect the new samples in a BTree set to ensure they stay
        // lexicographically ordered
        let mut merged_samples = BTreeSet::new();
        for (block, new_feature) in blocks_to_merge.iter().zip(new_feature_labels) {
            for sample in block.values.samples().iter() {
                merged_samples.insert(sample.to_vec());
            }

            let old_features = block.values.features();
            old_feature_sizes.push(old_features.count());
            for old_feature in old_features.iter() {
                let mut feature = new_feature.to_vec();
                feature.extend_from_slice(old_feature);
                new_features_builder.add(feature);
            }
        }

        let mut merged_samples_builder = LabelsBuilder::new(first_block.values.samples().names());
        for sample in merged_samples {
            merged_samples_builder.add(sample);
        }
        let merged_samples = merged_samples_builder.finish();

        // Vec<Vec<usize>> mapping from old values sample index (per block) to
        // the new sample index
        let mut samples_mapping = Vec::new();
        for block in &blocks_to_merge {
            let mut mapping_for_block = Vec::new();
            for sample in block.values.samples().iter() {
                let new_sample_i = merged_samples.position(sample).expect("missing entry in merged samples");
                mapping_for_block.push(new_sample_i);
            }
            samples_mapping.push(mapping_for_block);
        }

        let new_symmetric = Arc::clone(first_block.values.symmetric());
        let new_features = Arc::new(new_features_builder.finish());

        let new_shape = (
            merged_samples.count(),
            new_symmetric.count(),
            new_features.count(),
        );
        let mut new_data = first_block.values.data.create(new_shape);

        let mut feature_ranges = Vec::new();
        let mut start = 0;
        for size in old_feature_sizes {
            let stop = start + size;
            feature_ranges.push(start..stop);
            start = stop;
        }

        for ((block_i, block), feature_range) in blocks_to_merge.iter().enumerate().zip(&feature_ranges) {
            for sample_i in 0..block.values.samples().count() {
                let new_sample_i = samples_mapping[block_i][sample_i];
                new_data.set_from(
                    new_sample_i,
                    feature_range.clone(),
                    &block.values.data,
                    sample_i
                );
            }
        }

        let mut new_block = Block::new(new_data, merged_samples, new_symmetric, new_features).expect("constructed an invalid block");

        // now collect & merge the different gradients
        for gradient_name in first_block.gradients_list() {
            let new_gradient_samples = merge_gradient_samples(
                &blocks_to_merge, gradient_name, &samples_mapping
            );

            let mut new_gradient = first_block.values.data.create((
                new_gradient_samples.count(), new_shape.1, new_shape.2
            ));

            for ((block_i, block), feature_range) in blocks_to_merge.iter().enumerate().zip(&feature_ranges) {
                let gradient = block.get_gradient(gradient_name).expect("missing gradient");
                for (sample_i, grad_sample) in gradient.samples().iter().enumerate() {
                    // translate from the old sample id in gradients to the new ones
                    let mut grad_sample = grad_sample.to_vec();
                    let old_sample_i = grad_sample[0].usize();
                    grad_sample[0] = LabelValue::from(samples_mapping[block_i][old_sample_i]);

                    let new_sample_i = new_gradient_samples.position(&grad_sample).expect("missing entry in merged samples");
                    new_gradient.set_from(
                        new_sample_i,
                        feature_range.clone(),
                        &gradient.data,
                        sample_i
                    );
                }
            }

            new_block.add_gradient(gradient_name, new_gradient_samples, new_gradient).expect("created invalid gradients");
        }

        return Ok(new_block);
    }

    // TODO: variables?
    pub fn symmetric_to_features(&mut self) -> Result<(), Error> {
        for block in &self.blocks {
            if !block.gradients_list().is_empty() {
                unimplemented!("symmetric_to_features with gradients is not implemented yet")
            }
        }

        let old_blocks = std::mem::take(&mut self.blocks);
        let mut new_blocks = Vec::new();

        for block in old_blocks {
            let mut features_names = block.values.symmetric().names();
            features_names.extend_from_slice(&block.values.features().names());

            let mut new_features = LabelsBuilder::new(features_names);
            for symmetric in block.values.symmetric().iter() {
                for feature in block.values.features().iter() {
                    let mut new_feature = symmetric.to_vec();
                    new_feature.extend_from_slice(feature);

                    new_features.add(new_feature);
                }
            }
            let new_features = new_features.finish();

            let new_shape = (block.values.samples().count(), 1, new_features.count());

            let mut data = block.values.data;
            data.reshape(new_shape);

            new_blocks.push(Block::new(
                data,
                block.values.samples,
                Arc::new(Labels::single()),
                Arc::new(new_features),
            )?);
        }

        self.blocks = new_blocks;

        Ok(())
    }

    /// Move the given variables from the sparse labels to the sample labels of
    /// the blocks.
    ///
    /// The current blocks will be merged together according to the sparse
    /// labels remaining after removing `variables`. The resulting merged blocks
    /// will have `variables` as the last sample variables, preceded by the
    /// current samples.
    ///
    /// Currently, this function only works if all merged block have the same
    /// feature labels.
    pub fn sparse_to_samples(&mut self, variables: Vec<&str>) -> Result<(), Error> {
        // TODO: requested values
        // TODO: sparse_to_samples_no_gradients?

        if variables.is_empty() {
            return Ok(());
        }

        let (new_sparse, new_samples) = self.split_sparse_label(variables)?;

        let mut new_blocks = Vec::new();
        if new_sparse.count() == 1 {
            // create a single block with everything
            let mut matching = Vec::new();
            for i in 0..self.blocks.len() {
                matching.push(i);
            }

            let block = self.merge_blocks_along_samples(&matching, &new_samples)?;
            new_blocks.push(block);
        } else {
            for entry in new_sparse.iter() {
                let mut selection = LabelsBuilder::new(new_sparse.names());
                selection.add(entry.to_vec());

                let matching = self.find_matching_blocks(&selection.finish())?;
                new_blocks.push(self.merge_blocks_along_samples(&matching, &new_samples)?);
            }
        }


        self.sparse = new_sparse;
        self.blocks = new_blocks;

        return Ok(());
    }

    /// Merge the blocks with the given `block_idx` along the sample axis. The
    /// new sample names & values to add to the sample axis are passed in
    /// `new_sample_labels`.
    fn merge_blocks_along_samples(&self,
        block_idx: &[usize],
        new_sample_labels: &Labels,
    ) -> Result<Block, Error> {
        assert!(!block_idx.is_empty());

        let first_block = &self.blocks[block_idx[0]];
        let first_symmetric_label = first_block.values.symmetric();
        let first_features_label = first_block.values.features();

        let blocks_to_merge = block_idx.iter().map(|&i| &self.blocks[i]).collect::<Vec<_>>();
        for block in &blocks_to_merge {
            if block.values.symmetric() != first_symmetric_label {
                return Err(Error::InvalidParameter(
                    "can not move sparse label to samples if the blocks have \
                    different symmetric labels, call symmetric_to_features first".into()
                ))
            }

            if block.values.features() != first_features_label {
                return Err(Error::InvalidParameter(
                    "can not move sparse label to samples if the blocks have \
                    different feature labels".into() // TODO: this might be possible
                ))
            }
        }

        // we need to collect the new samples in a BTree set to ensure they stay
        // lexicographically ordered
        let mut merged_samples = BTreeSet::new();
        for (block, new_sample_label) in blocks_to_merge.iter().zip(new_sample_labels) {
            for old_sample in block.values.samples().iter() {
                let mut sample = old_sample.to_vec();
                sample.extend_from_slice(new_sample_label);
                merged_samples.insert(sample);
            }
        }

        let new_samples_names = first_block.values.samples().names().iter().copied()
            .chain(new_sample_labels.names())
            .collect();

        let mut merged_samples_builder = LabelsBuilder::new(new_samples_names);
        for sample in merged_samples {
            merged_samples_builder.add(sample);
        }
        let merged_samples = merged_samples_builder.finish();
        let new_symmetric = Arc::clone(first_block.values.symmetric());
        let new_features = Arc::clone(first_block.values.features());

        let new_shape = (
            merged_samples.count(),
            new_symmetric.count(),
            new_features.count(),
        );
        let mut new_data = first_block.values.data.create(new_shape);

        let mut samples_mapping = Vec::new();
        for (block, new_sample_label) in blocks_to_merge.iter().zip(new_sample_labels) {
            let mut mapping_for_block = Vec::new();
            for sample in block.values.samples().iter() {
                let mut new_sample = sample.to_vec();
                new_sample.extend_from_slice(new_sample_label);

                let new_sample_i = merged_samples.position(&new_sample).expect("missing entry in merged samples");
                mapping_for_block.push(new_sample_i);
            }
            samples_mapping.push(mapping_for_block);
        }

        let feature_range = 0..new_features.count();

        for (block_i, block) in blocks_to_merge.iter().enumerate() {
            for sample_i in 0..block.values.samples().count() {

                new_data.set_from(
                    samples_mapping[block_i][sample_i],
                    feature_range.clone(),
                    &block.values.data,
                    sample_i
                );
            }
        }

        let mut new_block = Block::new(new_data, merged_samples, new_symmetric, new_features).expect("invalid block");

        // now collect & merge the different gradients
        for gradient_name in first_block.gradients_list() {
            let new_gradient_samples = merge_gradient_samples(
                &blocks_to_merge, gradient_name, &samples_mapping
            );

            let mut new_gradient = first_block.values.data.create((
                new_gradient_samples.count(), new_shape.1, new_shape.2
            ));

            for (block_i, block) in blocks_to_merge.iter().enumerate() {
                let gradient = block.get_gradient(gradient_name).expect("missing gradient");
                for (sample_i, grad_sample) in gradient.samples().iter().enumerate() {
                    // translate from the old sample id in gradients to the new ones
                    let mut grad_sample = grad_sample.to_vec();
                    let old_sample_i = grad_sample[0].usize();
                    grad_sample[0] = LabelValue::from(samples_mapping[block_i][old_sample_i]);

                    let new_sample_i = new_gradient_samples.position(&grad_sample).expect("missing entry in merged samples");
                    new_gradient.set_from(
                        new_sample_i,
                        feature_range.clone(),
                        &gradient.data,
                        sample_i
                    );
                }
            }

            new_block.add_gradient(gradient_name, new_gradient_samples, new_gradient).expect("created invalid gradients");
        }

        return Ok(new_block);
    }
}

fn merge_gradient_samples(blocks: &[&Block], gradient_name: &str, mapping: &[Vec<usize>]) -> Labels {
    let mut new_gradient_samples = BTreeSet::new();
    let mut new_gradient_samples_names = None;
    for (block_i, block) in blocks.iter().enumerate() {
        let gradient = block.get_gradient(gradient_name).expect("missing gradient");

        if new_gradient_samples_names.is_none() {
            new_gradient_samples_names = Some(gradient.samples().names());
        }

        for grad_sample in gradient.samples().iter() {
            // translate from the old sample id in gradients to the new ones
            let mut grad_sample = grad_sample.to_vec();
            let old_sample_i = grad_sample[0].usize();
            grad_sample[0] = LabelValue::from(mapping[block_i][old_sample_i]);

            new_gradient_samples.insert(grad_sample);
        }
    }

    let mut new_gradient_samples_builder = LabelsBuilder::new(new_gradient_samples_names.expect("missing gradient samples names"));
    for sample in new_gradient_samples {
        new_gradient_samples_builder.add(sample);
    }
    return new_gradient_samples_builder.finish();
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::aml_array_t;
    use crate::data::TestArray;

    #[test]
    fn descriptor_validation() {
        let mut sparse = LabelsBuilder::new(vec!["sparse_1", "sparse_2"]);
        sparse.add(vec![LabelValue::new(0), LabelValue::new(0)]);
        sparse.add(vec![LabelValue::new(1), LabelValue::new(0)]);
        let sparse = sparse.finish();

        let mut samples_1 = LabelsBuilder::new(vec!["sample"]);
        samples_1.add(vec![LabelValue::new(0)]);
        let samples_1 = samples_1.finish();

        let mut samples_2 = LabelsBuilder::new(vec!["sample"]);
        samples_2.add(vec![LabelValue::new(33)]);
        samples_2.add(vec![LabelValue::new(-23)]);
        let samples_2 = samples_2.finish();

        let mut symmetric = LabelsBuilder::new(vec!["symmetric"]);
        symmetric.add(vec![LabelValue::new(0)]);
        let symmetric = Arc::new(symmetric.finish());

        let mut features = LabelsBuilder::new(vec!["features"]);
        features.add(vec![LabelValue::new(0)]);
        let features = Arc::new(features.finish());

        let block_1 = Block::new(
            aml_array_t::new(Box::new(TestArray::new((1, 1, 1)))),
            samples_1.clone(),
            Arc::clone(&symmetric),
            Arc::clone(&features),
        ).unwrap();

        let block_2 = Block::new(
            aml_array_t::new(Box::new(TestArray::new((2, 1, 1)))),
            samples_2.clone(),
            Arc::clone(&symmetric),
            Arc::clone(&features),
        ).unwrap();

        let result = Descriptor::new(sparse.clone(), vec![block_1, block_2]);
        assert!(result.is_ok());

        /**********************************************************************/
        let mut wrong_samples = LabelsBuilder::new(vec!["something_else"]);
        wrong_samples.add(vec![LabelValue::new(3)]);
        wrong_samples.add(vec![LabelValue::new(4)]);
        let wrong_samples = wrong_samples.finish();

        let block_1 = Block::new(
            aml_array_t::new(Box::new(TestArray::new((1, 1, 1)))),
            samples_1.clone(),
            Arc::clone(&symmetric),
            Arc::clone(&features),
        ).unwrap();

        let block_2 = Block::new(
            aml_array_t::new(Box::new(TestArray::new((2, 1, 1)))),
            wrong_samples,
            Arc::clone(&symmetric),
            Arc::clone(&features),
        ).unwrap();

        let error = Descriptor::new(sparse.clone(), vec![block_1, block_2]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid parameter: all blocks must have the same samples labels \
            names, got [something_else] and [sample]"
        );

        /**********************************************************************/
        let mut wrong_symmetric = LabelsBuilder::new(vec!["something_else"]);
        wrong_symmetric.add(vec![LabelValue::new(3)]);
        let wrong_symmetric = wrong_symmetric.finish();

        let block_1 = Block::new(
            aml_array_t::new(Box::new(TestArray::new((1, 1, 1)))),
            samples_1.clone(),
            Arc::clone(&symmetric),
            Arc::clone(&features),
        ).unwrap();

        let block_2 = Block::new(
            aml_array_t::new(Box::new(TestArray::new((2, 1, 1)))),
            samples_2.clone(),
            Arc::new(wrong_symmetric),
            Arc::clone(&features),
        ).unwrap();

        let error = Descriptor::new(sparse.clone(), vec![block_1, block_2]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid parameter: all blocks must have the same symmetric labels \
            names, got [something_else] and [symmetric]"
        );

        /**********************************************************************/
        let mut wrong_features = LabelsBuilder::new(vec!["something_else"]);
        wrong_features.add(vec![LabelValue::new(3)]);
        let wrong_features = wrong_features.finish();

        let block_1 = Block::new(
            aml_array_t::new(Box::new(TestArray::new((1, 1, 1)))),
            samples_1,
            Arc::clone(&symmetric),
            Arc::clone(&features),
        ).unwrap();

        let block_2 = Block::new(
            aml_array_t::new(Box::new(TestArray::new((2, 1, 1)))),
            samples_2,
            Arc::clone(&symmetric),
            Arc::new(wrong_features),
        ).unwrap();

        let error = Descriptor::new(sparse, vec![block_1, block_2]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid parameter: all blocks must have the same features labels \
            names, got [something_else] and [features]"
        );

        // TODO: check error messages for gradients
    }

    #[cfg(feature = "ndarray")]
    mod moving_labels {
        use super::*;
        use ndarray::{array, Array3};

        #[allow(clippy::too_many_lines)]
        fn example_descriptor() -> Descriptor {
            let mut samples_1 = LabelsBuilder::new(vec!["sample"]);
            samples_1.add(vec![LabelValue::new(0)]);
            samples_1.add(vec![LabelValue::new(2)]);
            samples_1.add(vec![LabelValue::new(4)]);
            let samples_1 = samples_1.finish();

            let mut symmetric_1 = LabelsBuilder::new(vec!["symmetric"]);
            symmetric_1.add(vec![LabelValue::new(0)]);
            let symmetric_1 = Arc::new(symmetric_1.finish());

            let mut features_1 = LabelsBuilder::new(vec!["features"]);
            features_1.add(vec![LabelValue::new(0)]);
            let features_1 = Arc::new(features_1.finish());

            let mut block_1 = Block::new(
                aml_array_t::new(Box::new(Array3::from_elem((3, 1, 1), 1.0))),
                samples_1,
                Arc::clone(&symmetric_1),
                Arc::clone(&features_1),
            ).unwrap();

            let mut gradient_samples_1 = LabelsBuilder::new(vec!["sample", "parameter"]);
            gradient_samples_1.add(vec![LabelValue::new(0), LabelValue::new(-2)]);
            gradient_samples_1.add(vec![LabelValue::new(2), LabelValue::new(3)]);
            let gradient_samples_1 = gradient_samples_1.finish();

            block_1.add_gradient(
                "parameter",
                gradient_samples_1,
                aml_array_t::new(Box::new(Array3::from_elem((2, 1, 1), 11.0)))
            ).unwrap();

            /******************************************************************/

            let mut samples_2 = LabelsBuilder::new(vec!["sample"]);
            samples_2.add(vec![LabelValue::new(0)]);
            samples_2.add(vec![LabelValue::new(1)]);
            samples_2.add(vec![LabelValue::new(3)]);
            let samples_2 = samples_2.finish();

            // different feature size
            let mut features_2 = LabelsBuilder::new(vec!["features"]);
            features_2.add(vec![LabelValue::new(3)]);
            features_2.add(vec![LabelValue::new(4)]);
            features_2.add(vec![LabelValue::new(5)]);
            let features_2 = Arc::new(features_2.finish());

            let mut block_2 = Block::new(
                aml_array_t::new(Box::new(Array3::from_elem((3, 1, 3), 2.0))),
                samples_2,
                symmetric_1,
                features_2,
            ).unwrap();

            let mut gradient_samples_2 = LabelsBuilder::new(vec!["sample", "parameter"]);
            gradient_samples_2.add(vec![LabelValue::new(0), LabelValue::new(-2)]);
            gradient_samples_2.add(vec![LabelValue::new(0), LabelValue::new(3)]);
            gradient_samples_2.add(vec![LabelValue::new(2), LabelValue::new(-2)]);
            let gradient_samples_2 = gradient_samples_2.finish();

            block_2.add_gradient(
                "parameter",
                gradient_samples_2,
                aml_array_t::new(Box::new(Array3::from_elem((3, 1, 3), 12.0)))
            ).unwrap();

            /******************************************************************/

            let mut samples_3 = LabelsBuilder::new(vec!["sample"]);
            samples_3.add(vec![LabelValue::new(0)]);
            samples_3.add(vec![LabelValue::new(3)]);
            samples_3.add(vec![LabelValue::new(6)]);
            samples_3.add(vec![LabelValue::new(8)]);
            let samples_3 = samples_3.finish();

            // different symmetric size
            let mut symmetric_2 = LabelsBuilder::new(vec!["symmetric"]);
            symmetric_2.add(vec![LabelValue::new(0)]);
            symmetric_2.add(vec![LabelValue::new(1)]);
            symmetric_2.add(vec![LabelValue::new(2)]);
            let symmetric_2 = Arc::new(symmetric_2.finish());

            let mut block_3 = Block::new(
                aml_array_t::new(Box::new(Array3::from_elem((4, 3, 1), 3.0))),
                samples_3,
                Arc::clone(&symmetric_2),
                Arc::clone(&features_1),
            ).unwrap();

            let mut gradient_samples_3 = LabelsBuilder::new(vec!["sample", "parameter"]);
            gradient_samples_3.add(vec![LabelValue::new(1), LabelValue::new(-2)]);
            let gradient_samples_3 = gradient_samples_3.finish();

            block_3.add_gradient(
                "parameter",
                gradient_samples_3,
                aml_array_t::new(Box::new(Array3::from_elem((1, 3, 1), 13.0)))
            ).unwrap();

            /******************************************************************/

            let mut samples_4 = LabelsBuilder::new(vec!["sample"]);
            samples_4.add(vec![LabelValue::new(0)]);
            samples_4.add(vec![LabelValue::new(1)]);
            samples_4.add(vec![LabelValue::new(2)]);
            samples_4.add(vec![LabelValue::new(5)]);
            let samples_4 = samples_4.finish();

            let mut block_4 = Block::new(
                aml_array_t::new(Box::new(Array3::from_elem((4, 3, 1), 4.0))),
                samples_4,
                symmetric_2,
                features_1,
            ).unwrap();

            let mut gradient_samples_4 = LabelsBuilder::new(vec!["sample", "parameter"]);
            gradient_samples_4.add(vec![LabelValue::new(0), LabelValue::new(1)]);
            gradient_samples_4.add(vec![LabelValue::new(3), LabelValue::new(3)]);
            let gradient_samples_4 = gradient_samples_4.finish();

            block_4.add_gradient(
                "parameter",
                gradient_samples_4,
                aml_array_t::new(Box::new(Array3::from_elem((2, 3, 1), 14.0)))
            ).unwrap();

            /******************************************************************/

            let mut sparse = LabelsBuilder::new(vec!["sparse_1", "sparse_2"]);
            sparse.add(vec![LabelValue::new(0), LabelValue::new(0)]);
            sparse.add(vec![LabelValue::new(1), LabelValue::new(0)]);
            sparse.add(vec![LabelValue::new(2), LabelValue::new(2)]);
            sparse.add(vec![LabelValue::new(2), LabelValue::new(3)]);
            let sparse = sparse.finish();

            return Descriptor::new(sparse, vec![block_1, block_2, block_3, block_4]).unwrap();
        }

        #[test]
        fn sparse_to_features() {
            let mut descriptor = example_descriptor();
            descriptor.sparse_to_features(vec!["sparse_1"]).unwrap();

            assert_eq!(descriptor.sparse().count(), 3);
            assert_eq!(descriptor.sparse().names(), ["sparse_2"]);
            assert_eq!(descriptor.sparse()[0], [LabelValue::new(0)]);
            assert_eq!(descriptor.sparse()[1], [LabelValue::new(2)]);
            assert_eq!(descriptor.sparse()[2], [LabelValue::new(3)]);

            assert_eq!(descriptor.blocks().len(), 3);

            // The new first block contains the old first two blocks merged
            let block_1 = &descriptor.blocks()[0];
            assert_eq!(block_1.values.samples().names(), ["sample"]);
            assert_eq!(block_1.values.samples().count(), 5);
            assert_eq!(block_1.values.samples()[0], [LabelValue::new(0)]);
            assert_eq!(block_1.values.samples()[1], [LabelValue::new(1)]);
            assert_eq!(block_1.values.samples()[2], [LabelValue::new(2)]);
            assert_eq!(block_1.values.samples()[3], [LabelValue::new(3)]);
            assert_eq!(block_1.values.samples()[4], [LabelValue::new(4)]);

            assert_eq!(block_1.values.symmetric().names(), ["symmetric"]);
            assert_eq!(block_1.values.symmetric().count(), 1);
            assert_eq!(block_1.values.symmetric()[0], [LabelValue::new(0)]);

            assert_eq!(block_1.values.features().names(), ["sparse_1", "features"]);
            assert_eq!(block_1.values.features().count(), 4);
            assert_eq!(block_1.values.features()[0], [LabelValue::new(0), LabelValue::new(0)]);
            assert_eq!(block_1.values.features()[1], [LabelValue::new(1), LabelValue::new(3)]);
            assert_eq!(block_1.values.features()[2], [LabelValue::new(1), LabelValue::new(4)]);
            assert_eq!(block_1.values.features()[3], [LabelValue::new(1), LabelValue::new(5)]);

            assert_eq!(block_1.values.data.as_array(), array![
                [[1.0, 2.0, 2.0, 2.0]],
                [[0.0, 2.0, 2.0, 2.0]],
                [[1.0, 0.0, 0.0, 0.0]],
                [[0.0, 2.0, 2.0, 2.0]],
                [[1.0, 0.0, 0.0, 0.0]],
            ]);

            let gradient_1 = block_1.get_gradient("parameter").unwrap();
            assert_eq!(gradient_1.samples().names(), ["sample", "parameter"]);
            assert_eq!(gradient_1.samples().count(), 4);
            assert_eq!(gradient_1.samples()[0], [LabelValue::new(0), LabelValue::new(-2)]);
            assert_eq!(gradient_1.samples()[1], [LabelValue::new(0), LabelValue::new(3)]);
            assert_eq!(gradient_1.samples()[2], [LabelValue::new(3), LabelValue::new(-2)]);
            assert_eq!(gradient_1.samples()[3], [LabelValue::new(4), LabelValue::new(3)]);

            assert_eq!(gradient_1.data.as_array(), array![
                [[11.0, 12.0, 12.0, 12.0]],
                [[0.0, 12.0, 12.0, 12.0]],
                [[0.0, 12.0, 12.0, 12.0]],
                [[11.0, 0.0, 0.0, 0.0]],
            ]);

            // The new second block contains the old third block
            let block_2 = &descriptor.blocks()[1];
            assert_eq!(block_2.values.data.shape(), (4, 3, 1));
            assert_eq!(block_2.values.data.as_array(), array![
                [[3.0], [3.0], [3.0]],
                [[3.0], [3.0], [3.0]],
                [[3.0], [3.0], [3.0]],
                [[3.0], [3.0], [3.0]],
            ]);

            // The new third block contains the old second block
            let block_3 = &descriptor.blocks()[2];
            assert_eq!(block_3.values.data.shape(), (4, 3, 1));
            assert_eq!(block_3.values.data.as_array(), array![
                [[4.0], [4.0], [4.0]],
                [[4.0], [4.0], [4.0]],
                [[4.0], [4.0], [4.0]],
                [[4.0], [4.0], [4.0]],
            ]);
        }

        #[test]
        fn sparse_to_samples() {
            let mut descriptor = example_descriptor();
            descriptor.sparse_to_samples(vec!["sparse_2"]).unwrap();

            assert_eq!(descriptor.sparse().count(), 3);
            assert_eq!(descriptor.sparse().names(), ["sparse_1"]);
            assert_eq!(descriptor.sparse()[0], [LabelValue::new(0)]);
            assert_eq!(descriptor.sparse()[1], [LabelValue::new(1)]);
            assert_eq!(descriptor.sparse()[2], [LabelValue::new(2)]);

            assert_eq!(descriptor.blocks().len(), 3);

            // The first two blocks are not modified
            let block_1 = &descriptor.blocks()[0];
            assert_eq!(block_1.values.data.shape(), (3, 1, 1));
            assert_eq!(block_1.values.data.as_array(), array![
                [[1.0]], [[1.0]], [[1.0]]
            ]);

            let block_2 = &descriptor.blocks()[1];
            assert_eq!(block_2.values.data.shape(), (3, 1, 3));
            assert_eq!(block_2.values.data.as_array(), array![
                [[2.0, 2.0, 2.0]],
                [[2.0, 2.0, 2.0]],
                [[2.0, 2.0, 2.0]],
            ]);

            // The new third block contains the old third and fourth blocks merged
            let block_3 = &descriptor.blocks()[2];
            assert_eq!(block_3.values.samples().names(), ["sample", "sparse_2"]);
            assert_eq!(block_3.values.samples().count(), 8);
            assert_eq!(block_3.values.samples()[0], [LabelValue::new(0), LabelValue::new(0)]);
            assert_eq!(block_3.values.samples()[1], [LabelValue::new(0), LabelValue::new(2)]);
            assert_eq!(block_3.values.samples()[2], [LabelValue::new(1), LabelValue::new(2)]);
            assert_eq!(block_3.values.samples()[3], [LabelValue::new(2), LabelValue::new(2)]);
            assert_eq!(block_3.values.samples()[4], [LabelValue::new(3), LabelValue::new(0)]);
            assert_eq!(block_3.values.samples()[5], [LabelValue::new(5), LabelValue::new(2)]);
            assert_eq!(block_3.values.samples()[6], [LabelValue::new(6), LabelValue::new(0)]);
            assert_eq!(block_3.values.samples()[7], [LabelValue::new(8), LabelValue::new(0)]);

            assert_eq!(block_3.values.symmetric().names(), ["symmetric"]);
            assert_eq!(block_3.values.symmetric().count(), 3);
            assert_eq!(block_3.values.symmetric()[0], [LabelValue::new(0)]);
            assert_eq!(block_3.values.symmetric()[1], [LabelValue::new(1)]);
            assert_eq!(block_3.values.symmetric()[2], [LabelValue::new(2)]);

            assert_eq!(block_3.values.features().names(), ["features"]);
            assert_eq!(block_3.values.features().count(), 1);
            assert_eq!(block_3.values.features()[0], [LabelValue::new(0)]);

            assert_eq!(block_3.values.data.as_array(), array![
                [[3.0], [3.0], [3.0]],
                [[4.0], [4.0], [4.0]],
                [[4.0], [4.0], [4.0]],
                [[4.0], [4.0], [4.0]],
                [[3.0], [3.0], [3.0]],
                [[4.0], [4.0], [4.0]],
                [[3.0], [3.0], [3.0]],
                [[3.0], [3.0], [3.0]],
            ]);

            let gradient_3 = block_3.get_gradient("parameter").unwrap();
            assert_eq!(gradient_3.samples().names(), ["sample", "parameter"]);
            assert_eq!(gradient_3.samples().count(), 3);
            assert_eq!(gradient_3.samples()[0], [LabelValue::new(1), LabelValue::new(1)]);
            assert_eq!(gradient_3.samples()[1], [LabelValue::new(4), LabelValue::new(-2)]);
            assert_eq!(gradient_3.samples()[2], [LabelValue::new(5), LabelValue::new(3)]);

            assert_eq!(gradient_3.data.as_array(), array![
                [[14.0], [14.0], [14.0]],
                [[13.0], [13.0], [13.0]],
                [[14.0], [14.0], [14.0]],
            ]);
        }

        #[test]
        fn symmetric_to_features() {
            // TODO
        }
    }
}
