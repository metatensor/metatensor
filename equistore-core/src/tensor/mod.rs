use std::collections::HashMap;
use std::sync::Arc;

use crate::{TensorBlock, BasicBlock};
use crate::{Labels, Error};

mod utils;

mod keys_to_samples;
mod keys_to_properties;


/// A tensor map is the main user-facing struct of this library, and can store
/// any kind of data used in atomistic machine learning.
///
/// A tensor map contains a list of `TensorBlock`s, each one associated with a
/// key in the form of a single `Labels` entry.
///
/// It provides functions to merge blocks together by moving some of these keys
/// to the samples or properties labels of the blocks, transforming the sparse
/// representation of the data to a dense one.
#[derive(Debug)]
pub struct TensorMap {
    keys: Arc<Labels>,
    blocks: Vec<TensorBlock>,
    // TODO: arbitrary tensor-level metadata? e.g. using `HashMap<String, String>`
}

fn check_labels_names(
    block: &BasicBlock,
    sample_names: &[&str],
    components_names: &[Vec<&str>],
    context: &str,
) -> Result<(), Error> {
    if block.samples.names() != sample_names {
        return Err(Error::InvalidParameter(format!(
            "all blocks must have the same sample label names, got [{}] and [{}]{}",
            block.samples.names().join(", "),
            sample_names.join(", "),
            context,
        )));
    }

    if block.components.len() != components_names.len() {
        return Err(Error::InvalidParameter(format!(
            "all blocks must contains the same set of components, the current \
            block has {} components while the first block has {}{}",
            block.components.len(),
            components_names.len(),
            context,
        )));
    }

    for (component_i, component) in block.components.iter().enumerate() {
        if component.names() != components_names[component_i] {
            return Err(Error::InvalidParameter(format!(
                "all blocks must have the same component label names, got [{}] and [{}]{}",
                component.names().join(", "),
                components_names[component_i].join(", "),
                context,
            )));
        }
    }

    Ok(())
}

impl TensorMap {
    /// Create a new `TensorMap` with the given keys and blocks.
    ///
    /// The number of keys must match the number of blocks, and all the blocks
    /// must contain the same kind of data (same labels names, same gradients
    /// defined on all blocks).
    #[allow(clippy::similar_names)]
    pub fn new(keys: Labels, blocks: Vec<TensorBlock>) -> Result<TensorMap, Error> {
        if blocks.len() != keys.count() {
            return Err(Error::InvalidParameter(format!(
                "expected the same number of blocks as the number of \
                entries in the keys ({}) when creating a tensor, got {}",
                keys.count(), blocks.len()
            )))
        }

        if !blocks.is_empty() {
            // make sure all blocks have the same kind of samples, components &
            // properties labels
            let sample_names = blocks[0].values().samples.names();
            let components_names = blocks[0].values().components.iter()
                .map(|c| c.names())
                .collect::<Vec<_>>();
            let properties_names = blocks[0].values().properties.names();

            let gradients_data = blocks[0].gradients().iter()
                .map(|(name, gradient)| {
                    let components_names = gradient.components.iter()
                        .map(|c| c.names())
                        .collect::<Vec<_>>();
                    (&**name, (gradient.samples.names(), components_names))
                })
                .collect::<HashMap<_, _>>();


            for block in &blocks {
                check_labels_names(block.values(), &sample_names, &components_names, "")?;

                if block.values().properties.names() != properties_names {
                    return Err(Error::InvalidParameter(format!(
                        "all blocks must have the same property label names, got [{}] and [{}]",
                        block.values().properties.names().join(", "),
                        properties_names.join(", "),
                    )));
                }

                if block.gradients().len() != gradients_data.len() {
                    return Err(Error::InvalidParameter(
                        "all blocks must contains the same set of gradients".into(),
                    ));
                }

                for (parameter, gradient) in block.gradients() {
                    match gradients_data.get(&**parameter) {
                        None => {
                            return Err(Error::InvalidParameter(format!(
                                "missing gradient with respect to {} in one of the blocks",
                                parameter
                            )));
                        },
                        Some((sample_names, components_names)) => {
                            check_labels_names(
                                gradient,
                                sample_names,
                                components_names,
                                &format!(" for gradients with respect to {}", parameter),
                            )?;
                        }
                    }
                }
            }
        }

        Ok(TensorMap {
            keys: Arc::new(keys),
            blocks,
        })
    }

    /// Try to copy this `TensorMap`. This can fail if we are unable to copy the
    /// underlying `eqs_array_t` data array
    pub fn try_clone(&self) -> Result<TensorMap, Error> {
        let mut blocks = Vec::new();
        for block in &self.blocks {
            blocks.push(block.try_clone()?);
        }

        return Ok(TensorMap {
            keys: Arc::clone(&self.keys),
            blocks
        });
    }

    /// Get the list of blocks in this `TensorMap`
    pub fn blocks(&self) -> &[TensorBlock] {
        &self.blocks
    }

    /// Get mutable access to the list of blocks in this `TensorMap`
    pub fn blocks_mut(&mut self) -> &mut [TensorBlock] {
        &mut self.blocks
    }

    /// Get the keys defined in this `TensorMap`
    pub fn keys(&self) -> &Arc<Labels> {
        &self.keys
    }

    /// Get the index of blocks matching the given selection.
    ///
    /// The selection must contains a single entry, defining the requested key
    /// or keys. If the selection contains only a subset of the variables of the
    /// keys, there can be multiple matching blocks.
    pub fn blocks_matching(&self, selection: &Labels) -> Result<Vec<usize>, Error> {
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
            for (i, &name) in self.keys.names().iter().enumerate() {
                if requested == name {
                    variables.push(i);
                    continue 'outer;
                }
            }

            return Err(Error::InvalidParameter(format!(
                "'{}' is not part of the keys for this tensor",
                requested
            )));
        }

        let mut matching = Vec::new();
        let selection = selection.iter().next().expect("empty selection");

        for (block_i, labels) in self.keys.iter().enumerate() {
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

    /// Move the given variables from the component labels to the property labels
    /// for each block in this `TensorMap`.
    pub fn components_to_properties(&self, variables: &[&str]) -> Result<TensorMap, Error> {
        let mut clone = self.try_clone()?;

        if variables.is_empty() {
            return Ok(clone);
        }

        for block in &mut clone.blocks {
            block.components_to_properties(variables)?;
        }

        return Ok(clone);
    }
}


#[cfg(test)]
mod tests {
    use crate::LabelsBuilder;
    use crate::data::TestArray;

    use super::*;
    use super::utils::example_labels;

    #[test]
    #[allow(clippy::too_many_lines)]
    fn blocks_validation() {
        let block_1 = TensorBlock::new(
            TestArray::new(vec![1, 1, 1]),
            example_labels(vec!["samples"], vec![[0]]),
            vec![example_labels(vec!["components"], vec![[0]])],
            example_labels(vec!["properties"], vec![[0]]),
        ).unwrap();

        let block_2 = TensorBlock::new(
            TestArray::new(vec![2, 3, 1]),
            example_labels(vec!["samples"], vec![[0], [1]]),
            vec![example_labels(vec!["components"], vec![[0], [1], [2]])],
            example_labels(vec!["properties"], vec![[0]]),
        ).unwrap();

        let result = TensorMap::new(
            (*example_labels(vec!["keys"], vec![[0], [1]])).clone(),
            vec![block_1, block_2],
        );
        assert!(result.is_ok());

        /**********************************************************************/
        let block_1 = TensorBlock::new(
            TestArray::new(vec![1, 1]),
            example_labels(vec!["samples"], vec![[0]]),
            vec![],
            example_labels(vec!["properties"], vec![[0]]),
        ).unwrap();

        let block_2 = TensorBlock::new(
            TestArray::new(vec![2, 1]),
            example_labels(vec!["something_else"], vec![[0], [1]]),
            vec![],
            example_labels(vec!["properties"], vec![[0]]),
        ).unwrap();

        let result = TensorMap::new(
            (*example_labels(vec!["keys"], vec![[0], [1]])).clone(),
            vec![block_1, block_2],
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: all blocks must have the same sample label \
            names, got [something_else] and [samples]"
        );

        /**********************************************************************/
        let block_1 = TensorBlock::new(
            TestArray::new(vec![1, 1, 1]),
            example_labels(vec!["samples"], vec![[0]]),
            vec![example_labels(vec!["components"], vec![[0]])],
            example_labels(vec!["properties"], vec![[0]]),
        ).unwrap();

        let block_2 = TensorBlock::new(
            TestArray::new(vec![2, 1]),
            example_labels(vec!["samples"], vec![[0], [1]]),
            vec![],
            example_labels(vec!["properties"], vec![[0]]),
        ).unwrap();

        let result = TensorMap::new(
            (*example_labels(vec!["keys"], vec![[0], [1]])).clone(),
            vec![block_1, block_2],
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: all blocks must contains the same set of \
            components, the current block has 0 components while the first \
            block has 1"
        );

        /**********************************************************************/
        let block_1 = TensorBlock::new(
            TestArray::new(vec![1, 1, 1]),
            example_labels(vec!["samples"], vec![[0]]),
            vec![example_labels(vec!["components"], vec![[0]])],
            example_labels(vec!["properties"], vec![[0]]),
        ).unwrap();

        let block_2 = TensorBlock::new(
            TestArray::new(vec![2, 3, 1]),
            example_labels(vec!["samples"], vec![[0], [1]]),
            vec![example_labels(vec!["something_else"], vec![[0], [1], [2]])],
            example_labels(vec!["properties"], vec![[0]]),
        ).unwrap();

        let result = TensorMap::new(
            (*example_labels(vec!["keys"], vec![[0], [1]])).clone(),
            vec![block_1, block_2],
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: all blocks must have the same component label \
            names, got [something_else] and [components]"
        );

        /**********************************************************************/
        let block_1 = TensorBlock::new(
            TestArray::new(vec![1, 1]),
            example_labels(vec!["samples"], vec![[0]]),
            vec![],
            example_labels(vec!["properties"], vec![[0]]),
        ).unwrap();

        let block_2 = TensorBlock::new(
            TestArray::new(vec![2, 1]),
            example_labels(vec!["samples"], vec![[0], [1]]),
            vec![],
            example_labels(vec!["something_else"], vec![[0]]),
        ).unwrap();

        let result = TensorMap::new(
            (*example_labels(vec!["keys"], vec![[0], [1]])).clone(),
            vec![block_1, block_2],
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: all blocks must have the same property label \
            names, got [something_else] and [properties]"
        );

        // TODO: check error messages for gradients
    }

    #[test]
    fn blocks_matching() {
        let mut blocks = Vec::new();
        for _ in 0..6 {
            blocks.push(TensorBlock::new(
                TestArray::new(vec![1, 1]),
                example_labels(vec!["samples"], vec![[0]]),
                vec![],
                example_labels(vec!["properties"], vec![[0]]),
            ).unwrap());
        }

        let keys = example_labels(vec!["key_1", "key_2"], vec![
            [0, 1], [0, 2], [1, 1],
            [1, 2], [3, 0], [4, 3],
        ]);

        let tensor = TensorMap::new((*keys).clone(), blocks).unwrap();

        let mut selection = LabelsBuilder::new(vec!["key_1", "key_2"]);
        selection.add(&[1, 1]).unwrap();
        assert_eq!(
            tensor.blocks_matching(&selection.finish()).unwrap(),
            [2]
        );

        let mut selection = LabelsBuilder::new(vec!["key_1"]);
        selection.add(&[1]).unwrap();
        assert_eq!(
            tensor.blocks_matching(&selection.finish()).unwrap(),
            [2, 3]
        );

        let selection = LabelsBuilder::new(vec!["key_1"]);
        let result = tensor.blocks_matching(&selection.finish());
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: block selection labels must contain a single row, got 0"
        );

        let mut selection = LabelsBuilder::new(vec!["key_1", "key_2"]);
        selection.add(&[3, 4]).unwrap();
        selection.add(&[1, 2]).unwrap();
        let result = tensor.blocks_matching(&selection.finish());
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: block selection labels must contain a single row, got 2"
        );

        let mut selection = LabelsBuilder::new(vec!["key_3"]);
        selection.add(&[1]).unwrap();
        let result = tensor.blocks_matching(&selection.finish());
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: 'key_3' is not part of the keys for this tensor"
        );
    }
}
