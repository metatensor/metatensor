use std::collections::BTreeSet;
use std::sync::Arc;

use indexmap::IndexSet;

use crate::{Block, Error};
use crate::{Labels, LabelsBuilder, LabelValue};

pub struct Descriptor {
    sparse: Labels,
    blocks: Vec<Block>,
}

impl Descriptor {
    pub fn new(sparse: Labels, blocks: Vec<Block>) -> Result<Descriptor, Error> {
        if blocks.len() != sparse.count() {
            return Err(Error::InvalidParameter(format!(
                "TODO: error message, expected {}, got {}", sparse.count(), blocks.len()
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
            return Err(Error::InvalidParameter("TODO: too many matching blocks".into()));
        }
        return Ok(&self.blocks[matching[0]]);
    }

    pub fn block_mut(&mut self, selection: &Labels) -> Result<&mut Block, Error> {
        let matching = self.find_matching_blocks(selection)?;
        if matching.len() != 1 {
            return Err(Error::InvalidParameter("TODO: too many matching blocks".into()));
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

    // TODO: requested values in sparse_to_features
    // TODO: densify_no_gradients
    pub fn sparse_to_features(&mut self, variables: &[&str]) -> Result<(), Error> {
        if variables.is_empty() {
            return Ok(());
        }

        let (new_sparse, new_features) = self.split_sparse_label(variables);

        let mut new_blocks = Vec::new();
        if new_sparse.count() == 1 {
            // create a single block with everything
            let mut matching = Vec::new();
            for i in 0..self.blocks.len() {
                matching.push(i);
            }

            let block = self.merge_blocks_along_features(&matching, variables, &new_features)?;
            new_blocks.push(block);
        } else {
            for entry in new_sparse.iter() {
                let mut selection = LabelsBuilder::new(new_sparse.names());
                selection.add(entry.to_vec());

                let matching = self.find_matching_blocks(&selection.finish())?;
                new_blocks.push(self.merge_blocks_along_features(&matching, variables, &new_features)?);
            }
        }


        self.sparse = new_sparse;
        self.blocks = new_blocks;

        return Ok(());
    }

    fn split_sparse_label(&self, variables: &[&str]) -> (Labels, Vec<Vec<LabelValue>>) {
        let mut remaining = Vec::new();
        let mut remaining_i = Vec::new();
        let mut variables_i = Vec::new();

        let sparse_names = self.sparse.names();
        'outer: for (i, &name) in sparse_names.iter().enumerate() {
            for &variable in variables {
                if variable == name {
                    variables_i.push(i);
                    continue 'outer;
                }
            }
            remaining.push(name);
            remaining_i.push(i);
        }

        // split_labels contains the values taken by the variables in the
        // sparse labels
        let mut split_labels = IndexSet::new();
        let mut new_sparse = IndexSet::new();
        for entry in self.sparse.iter() {
            let mut label = Vec::new();
            for &i in &variables_i {
                label.push(entry[i]);
            }
            split_labels.insert(label);

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

        return (new_sparse, split_labels.into_iter().collect());
    }

    fn merge_blocks_along_features(&self,
        block_idx: &[usize],
        feature_names: &[&str],
        feature_values: &[Vec<LabelValue>],
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

            if self.blocks[id].has_gradients() {
                unimplemented!("sparse_to_features with gradients is not implemented yet")
            }
        }

        let new_feature_names = feature_names.iter()
            .chain(first_block.values.features().names().iter())
            .copied()
            .collect();
        let mut new_features = LabelsBuilder::new(new_feature_names);
        let mut old_feature_sizes = Vec::new();

        let mut new_samples = LabelsBuilder::new(first_block.values.samples().names());

        for (&id, new_feature) in block_idx.iter().zip(feature_values) {
            let block = &self.blocks[id];

            for sample in block.values.samples().iter() {
                if !new_samples.contains(sample) {
                    new_samples.add(sample.to_vec());
                }
            }

            let old_features = block.values.features();
            old_feature_sizes.push(old_features.count());
            for old_feature in old_features.iter() {
                let mut feature = new_feature.clone();
                feature.extend_from_slice(old_feature);
                new_features.add(feature);
            }
        }

        let new_samples = new_samples.finish();
        let new_symmetric = Arc::clone(first_block.values.symmetric());
        let new_features = Arc::new(new_features.finish());

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
                new_data.set_from_other(
                    new_sample_i,
                    feature_range.clone(),
                    &block.values.data,
                    sample_i
                );
            }
        }

        return Ok(Block::new(new_data, new_samples, new_symmetric, new_features));
    }

    // TODO: variables?
    pub fn symmetric_to_features(&mut self) -> Result<(), Error> {
        for block in &self.blocks {
            if block.has_gradients() {
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
            ));
        }

        self.blocks = new_blocks;

        Ok(())
    }

    // TODO: requested values
    // TODO: moving only values?
    pub fn sparse_to_samples(&mut self, variables: &[&str]) -> Result<(), Error> {
        if variables.is_empty() {
            return Ok(());
        }

        let (new_sparse, new_samples) = self.split_sparse_label(variables);

        let mut new_blocks = Vec::new();
        if new_sparse.count() == 1 {
            // create a single block with everything
            let mut matching = Vec::new();
            for i in 0..self.blocks.len() {
                matching.push(i);
            }

            let block = self.merge_blocks_along_samples(&matching, variables, &new_samples)?;
            new_blocks.push(block);
        } else {
            for entry in new_sparse.iter() {
                let mut selection = LabelsBuilder::new(new_sparse.names());
                selection.add(entry.to_vec());

                let matching = self.find_matching_blocks(&selection.finish())?;
                new_blocks.push(self.merge_blocks_along_samples(&matching, variables, &new_samples)?);
            }
        }


        self.sparse = new_sparse;
        self.blocks = new_blocks;

        return Ok(());
    }

    fn merge_blocks_along_samples(&self,
        block_idx: &[usize],
        samples_names: &[&str],
        samples_values: &[Vec<LabelValue>]
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

            if self.blocks[id].has_gradients() {
                unimplemented!("sparse_to_features with gradients is not implemented yet")
            }
        }

        // TODO: why a BTreeSet?
        let mut new_samples = BTreeSet::new();
        for (&id, new_sample_values) in block_idx.iter().zip(samples_values) {
            let block = &self.blocks[id];

            for old_sample in block.values.samples().iter() {
                let mut sample = old_sample.to_vec();
                sample.extend_from_slice(new_sample_values);
                new_samples.insert(sample);
            }
        }

        let new_samples_names = first_block.values.samples().names().iter()
            .chain(samples_names)
            .copied()
            .collect();
        let mut new_samples_builder = LabelsBuilder::new(new_samples_names);
        for sample in new_samples {
            new_samples_builder.add(sample);
        }
        let new_samples = new_samples_builder.finish();

        let new_symmetric = Arc::clone(first_block.values.symmetric());
        let new_features = Arc::clone(first_block.values.features());

        let new_shape = (
            new_samples.count(),
            new_symmetric.count(),
            new_features.count(),
        );
        let mut new_data = first_block.values.data.create(new_shape);

        let feature_range = 0..new_features.count();

        for (&id, new_sample_values) in block_idx.iter().zip(samples_values) {
            let block = &self.blocks[id];

            for (old_sample_i, old_sample) in block.values.samples().iter().enumerate() {
                let mut new_sample = old_sample.to_vec();
                new_sample.extend_from_slice(new_sample_values);

                let new_sample_i = new_samples.position(&new_sample).expect("missing entry in merged samples");
                new_data.set_from_other(
                    new_sample_i,
                    feature_range.clone(),
                    &block.values.data,
                    old_sample_i
                );
            }
        }

        return Ok(Block::new(new_data, new_samples, new_symmetric, new_features));
    }
}
