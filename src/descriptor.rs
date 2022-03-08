use std::collections::BTreeSet;
use std::sync::Arc;

use indexmap::IndexSet;

use crate::{Block, Error};
use crate::{Labels, LabelsBuilder, LabelValue};

#[derive(Debug)]
pub struct Descriptor {
    sparse: Labels,
    blocks: Vec<Block>,
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

                // TODO: gradients validation
            }
        }

        Ok(Descriptor {
            sparse,
            blocks,
        })
    }

    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    pub fn blocks_mut(&mut self) -> &mut [Block] {
        &mut self.blocks
    }

    pub fn sparse(&self) -> &Labels {
        &self.sparse
    }

    pub fn iter(&self) -> impl Iterator<Item=(&[LabelValue], &Block)> + '_ {
        self.sparse.iter().zip(&self.blocks)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=(&[LabelValue], &mut Block)> + '_ {
        self.sparse.iter().zip(&mut self.blocks)
    }

    pub fn blocks_matching(&self, selection: &Labels) -> Result<Vec<&Block>, Error> {
        let matching = self.find_matching_blocks(selection)?;

        return Ok(matching.into_iter().map(|i| &self.blocks[i]).collect());
    }

    pub fn blocks_matching_mut(&mut self, selection: &Labels) -> Result<Vec<&mut Block>, Error> {
        let _matching = self.find_matching_blocks(selection)?;

        // TODO: needs unsafe to tell compiler we are only giving out exclusive references
        // return matching.into_iter().map(|i| &'a mut self.blocks[i]).collect();
        todo!()
    }

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

    /// Move the given variables from the sparse labels to the feature labels.
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

        // TODO: check for unicity in variables

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

    /// Merge the blocks with the given `block_idx` along the feature axis.
    /// TODO
    fn merge_blocks_along_features(&self,
        block_idx: &[usize],
        new_features: &Labels,
    ) -> Result<Block, Error> {
        assert!(!block_idx.is_empty());

        let first_block = &self.blocks[block_idx[0]];
        let first_symmetric_label = first_block.values.symmetric();
        for &id in block_idx {
            if self.blocks[id].values.symmetric() != first_symmetric_label {
                return Err(Error::InvalidParameter(
                    "can not move sparse label to features if the blocks have \
                    different symmetric labels, call symmetric_to_features first".into()
                ))
            }

            if !self.blocks[id].gradients_list().is_empty() {
                unimplemented!("sparse_to_features with gradients is not implemented yet")
            }
        }

        let new_feature_names = new_features.names().iter()
            .chain(first_block.values.features().names().iter())
            .copied()
            .collect();
        let mut new_features_builder = LabelsBuilder::new(new_feature_names);
        let mut old_feature_sizes = Vec::new();

        // we need to collect the new samples in a BTree set to ensure they stay
        // lexicographically ordered
        let mut new_samples = BTreeSet::new();
        for (&id, new_feature) in block_idx.iter().zip(new_features) {
            let block = &self.blocks[id];

            for sample in block.values.samples().iter() {
                new_samples.insert(sample.to_vec());
            }

            let old_features = block.values.features();
            old_feature_sizes.push(old_features.count());
            for old_feature in old_features.iter() {
                let mut feature = new_feature.to_vec();
                feature.extend_from_slice(old_feature);
                new_features_builder.add(feature);
            }
        }

        let mut new_samples_builder = LabelsBuilder::new(first_block.values.samples().names());
        for sample in new_samples {
            new_samples_builder.add(sample);
        }
        let new_samples = new_samples_builder.finish();

        let new_symmetric = Arc::clone(first_block.values.symmetric());
        let new_features = Arc::new(new_features_builder.finish());

        let mut feature_ranges = Vec::new();
        let mut start = 0;
        for size in old_feature_sizes {
            let stop = start + size;
            feature_ranges.push(start..stop);
            start = stop;
        }

        let new_shape = (
            new_samples.count(),
            new_symmetric.count(),
            new_features.count(),
        );
        let mut new_data = first_block.values.data.create(new_shape);

        for (block, feature_range) in block_idx.iter().map(|&block_id| &self.blocks[block_id]).zip(feature_ranges) {
            for (sample_i, sample) in block.values.samples().iter().enumerate() {
                let new_sample_i = new_samples.position(sample).expect("missing entry in merged samples");
                new_data.set_from(
                    new_sample_i,
                    feature_range.clone(),
                    &block.values.data,
                    sample_i
                );
            }
        }

        return Block::new(new_data, new_samples, new_symmetric, new_features);
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

    // TODO: requested values
    // TODO: moving only values?
    pub fn sparse_to_samples(&mut self, variables: Vec<&str>) -> Result<(), Error> {
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

    fn merge_blocks_along_samples(&self,
        block_idx: &[usize],
        new_samples: &Labels,
    ) -> Result<Block, Error> {
        assert!(!block_idx.is_empty());

        let first_block = &self.blocks[block_idx[0]];
        let first_symmetric_label = first_block.values.symmetric();
        let first_features_label = first_block.values.features();
        for &id in block_idx {
            if self.blocks[id].values.symmetric() != first_symmetric_label {
                return Err(Error::InvalidParameter(
                    "can not move sparse label to samples if the blocks have \
                    different symmetric labels, call symmetric_to_features first".into()
                ))
            }

            if self.blocks[id].values.features() != first_features_label {
                return Err(Error::InvalidParameter(
                    "can not move sparse label to samples if the blocks have \
                    different feature labels".into() // TODO: this might be possible
                ))
            }

            if !self.blocks[id].gradients_list().is_empty() {
                unimplemented!("sparse_to_features with gradients is not implemented yet")
            }
        }

        // we need to collect the new samples in a BTree set to ensure they stay
        // lexicographically ordered
        let mut new_samples_set = BTreeSet::new();
        for (&id, new_sample_values) in block_idx.iter().zip(new_samples) {
            let block = &self.blocks[id];

            for old_sample in block.values.samples().iter() {
                let mut sample = old_sample.to_vec();
                sample.extend_from_slice(new_sample_values);
                new_samples_set.insert(sample);
            }
        }

        let new_symmetric = Arc::clone(first_block.values.symmetric());
        let new_features = Arc::clone(first_block.values.features());

        let new_shape = (
            new_samples_set.len(),
            new_symmetric.count(),
            new_features.count(),
        );
        let mut new_data = first_block.values.data.create(new_shape);

        let feature_range = 0..new_features.count();

        for (&id, new_sample_values) in block_idx.iter().zip(new_samples) {
            let block = &self.blocks[id];

            for (old_sample_i, old_sample) in block.values.samples().iter().enumerate() {
                let mut new_sample = old_sample.to_vec();
                new_sample.extend_from_slice(new_sample_values);

                let new_sample_i = new_samples.position(&new_sample).expect("missing entry in merged samples");
                new_data.set_from(
                    new_sample_i,
                    feature_range.clone(),
                    &block.values.data,
                    old_sample_i
                );
            }
        }

        let new_samples_names = first_block.values.samples().names().iter().copied()
            .chain(new_samples.names())
            .collect();
        let mut new_samples_builder = LabelsBuilder::new(new_samples_names);
        for sample in new_samples_set {
            new_samples_builder.add(sample);
        }
        let new_samples = new_samples_builder.finish();

        return Block::new(new_data, new_samples, new_symmetric, new_features);
    }
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

            let block_1 = Block::new(
                aml_array_t::new(Box::new(Array3::from_elem((3, 1, 1), 1.0))),
                samples_1,
                Arc::clone(&symmetric_1),
                Arc::clone(&features_1),
            ).unwrap();

            let mut samples_2 = LabelsBuilder::new(vec!["sample"]);
            samples_2.add(vec![LabelValue::new(1)]);
            samples_2.add(vec![LabelValue::new(3)]);
            samples_2.add(vec![LabelValue::new(5)]);
            let samples_2 = samples_2.finish();

            // different feature size
            let mut features_2 = LabelsBuilder::new(vec!["features"]);
            features_2.add(vec![LabelValue::new(3)]);
            features_2.add(vec![LabelValue::new(4)]);
            features_2.add(vec![LabelValue::new(5)]);
            let features_2 = Arc::new(features_2.finish());

            let block_2 = Block::new(
                aml_array_t::new(Box::new(Array3::from_elem((3, 1, 3), 2.0))),
                samples_2,
                symmetric_1,
                features_2,
            ).unwrap();

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

            let block_3 = Block::new(
                aml_array_t::new(Box::new(Array3::from_elem((4, 3, 1), 3.0))),
                samples_3,
                symmetric_2,
                features_1,
            ).unwrap();

            let mut sparse = LabelsBuilder::new(vec!["sparse_1", "sparse_2"]);
            sparse.add(vec![LabelValue::new(0), LabelValue::new(0)]);
            sparse.add(vec![LabelValue::new(1), LabelValue::new(0)]);
            sparse.add(vec![LabelValue::new(1), LabelValue::new(2)]);
            let sparse = sparse.finish();

            return Descriptor::new(sparse, vec![block_1, block_2, block_3]).unwrap();
        }

        #[test]
        fn sparse_to_features() {
            let mut descriptor = example_descriptor();
            descriptor.sparse_to_features(vec!["sparse_1"]).unwrap();

            assert_eq!(descriptor.sparse().count(), 2);
            assert_eq!(descriptor.sparse().names(), ["sparse_2"]);
            assert_eq!(descriptor.sparse()[0], [LabelValue::new(0)]);
            assert_eq!(descriptor.sparse()[1], [LabelValue::new(2)]);

            assert_eq!(descriptor.blocks().len(), 2);

            // The new first block contains the old first two blocks merged
            let block_1 = &descriptor.blocks()[0];
            assert_eq!(block_1.values.samples().names(), ["sample"]);
            assert_eq!(block_1.values.samples().count(), 6);
            // TODO: should re-order the samples?
            assert_eq!(block_1.values.samples()[0], [LabelValue::new(0)]);
            assert_eq!(block_1.values.samples()[1], [LabelValue::new(1)]);
            assert_eq!(block_1.values.samples()[2], [LabelValue::new(2)]);
            assert_eq!(block_1.values.samples()[3], [LabelValue::new(3)]);
            assert_eq!(block_1.values.samples()[4], [LabelValue::new(4)]);
            assert_eq!(block_1.values.samples()[5], [LabelValue::new(5)]);

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
                [[1.0, 0.0, 0.0, 0.0]],
                [[0.0, 2.0, 2.0, 2.0]],
                [[1.0, 0.0, 0.0, 0.0]],
                [[0.0, 2.0, 2.0, 2.0]],
                [[1.0, 0.0, 0.0, 0.0]],
                [[0.0, 2.0, 2.0, 2.0]]
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
        }

        #[test]
        fn sparse_to_samples() {
            // TODO
        }

        #[test]
        fn symmetric_to_features() {
            // TODO
        }
    }
}
