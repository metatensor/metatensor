use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::ffi::CString;

use crate::utils::ConstCString;
use crate::TensorBlock;
use crate::{Labels, Error};
use crate::get_data_origin;

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

    info: BTreeMap<String, ConstCString>,
    // all the keys from `info`, as C-compatible strings
    info_keys: Vec<ConstCString>
}

fn check_labels_dimensions(
    block: &TensorBlock,
    sample_dimensions: &[&str],
    component_dimensions: &[Vec<&str>],
    context: &str,
) -> Result<(), Error> {
    if block.samples.dimensions() != sample_dimensions {
        return Err(Error::InvalidParameter(format!(
            "all blocks must have the same sample dimensions, got [{}] and [{}]{}",
            block.samples.dimensions().join(", "),
            sample_dimensions.join(", "),
            context,
        )));
    }

    if block.components.len() != component_dimensions.len() {
        return Err(Error::InvalidParameter(format!(
            "all blocks must contains the same set of components, the current \
            block has {} components while the first block has {}{}",
            block.components.len(),
            component_dimensions.len(),
            context,
        )));
    }

    for (component_i, component) in block.components.iter().enumerate() {
        if component.dimensions() != component_dimensions[component_i] {
            return Err(Error::InvalidParameter(format!(
                "all blocks must have the same component dimensions, got [{}] and [{}]{}",
                component.dimensions().join(", "),
                component_dimensions[component_i].join(", "),
                context,
            )));
        }
    }

    Ok(())
}

fn check_origin(blocks: &[TensorBlock]) -> Result<(), Error> {
    if blocks.is_empty() {
        return Ok(());
    }

    let first_origin = blocks[0].values.origin()?;
    for block in blocks.iter().skip(1) {
        let block_origin = block.values.origin()?;
        if first_origin != block_origin {
            return Err(Error::InvalidParameter(format!(
                "tried to build a TensorMap from blocks with different origins: at least ('{}') and ('{}') were detected",
                get_data_origin(first_origin),
                get_data_origin(block_origin),
            )));
        }
    }

    Ok(())
}

fn check_device(blocks: &[TensorBlock]) -> Result<(), Error> {
    if blocks.is_empty() {
        return Ok(());
    }

    let first_device = blocks[0].values.device()?;
    for block in blocks.iter().skip(1) {
        let block_device = block.values.device()?;
        if first_device != block_device {
            return Err(Error::InvalidParameter(format!(
                "tried to build a TensorMap from blocks on different devices: at least '{}' and '{}' were detected",
                first_device, block_device,
            )));
        }
    }

    Ok(())
}

fn check_dtype(blocks: &[TensorBlock]) -> Result<(), Error> {
    if blocks.is_empty() {
        return Ok(());
    }

    let first_dtype = blocks[0].values.dtype()?;
    for block in blocks.iter().skip(1) {
        let block_dtype = block.values.dtype()?;
        if first_dtype != block_dtype {
            return Err(Error::InvalidParameter(format!(
                "tried to build a TensorMap from blocks with different dtypes: at least '{}' and '{}' were detected",
                first_dtype, block_dtype,
            )));
        }
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct GradientMetadata<'a> {
    sample_dimensions: Vec<&'a str>,
    component_dimensions: Vec<Vec<&'a str>>,
}

#[derive(Debug, Clone, PartialEq)]
struct GradientMap<'a> {
    // a struct that contains all the gradient information for a single
    // TensorBlock. The GradientMap is recursive for gradients of gradients
    gradients: HashMap<String, (GradientMetadata<'a>, GradientMap<'a>)>
}

impl GradientMap<'_> {
    fn new(block: &TensorBlock) -> GradientMap<'_> {
        // generate gradient information for a block
        let mut gradients = HashMap::new();
        for (gradient_name, sub_gradient) in block.gradients() {
            let metadata = GradientMetadata {
                sample_dimensions: sub_gradient.samples.dimensions(),
                component_dimensions: sub_gradient.components.iter()
                    .map(|c| c.dimensions())
                    .collect::<Vec<_>>(),
            };
            gradients.insert(gradient_name.clone(), (metadata, GradientMap::new(sub_gradient)));
        }
        GradientMap { gradients }
    }
}

impl TensorMap {
    /// Create a new `TensorMap` with the given keys and blocks.
    ///
    /// The number of keys must match the number of blocks, and all the blocks
    /// must contain the same kind of data (same labels names, same gradients
    /// defined on all blocks).
    #[allow(clippy::similar_names)]
    pub fn new(keys: Arc<Labels>, blocks: Vec<TensorBlock>) -> Result<TensorMap, Error> {
        if blocks.len() != keys.count() {
            return Err(Error::InvalidParameter(format!(
                "expected the same number of blocks as the number of \
                entries in the keys ({}) when creating a `TensorMap`, got {}",
                keys.count(), blocks.len()
            )))
        }

        check_origin(&blocks)?;
        check_device(&blocks)?;
        check_dtype(&blocks)?;

        if !blocks.is_empty() {
            // extract metadata from the first block
            let sample_dimensions = blocks[0].samples.dimensions();
            let component_dimensions = blocks[0].components.iter()
                .map(|c| c.dimensions())
                .collect::<Vec<_>>();
            let property_dimensions = blocks[0].properties.dimensions();
            let gradient_map = GradientMap::new(&blocks[0]);

            for block in &blocks {
                // check samples and components are the same as those of the first block
                check_labels_dimensions(block, &sample_dimensions, &component_dimensions, "")?;

                // check properties are the same as those of the first block
                if block.properties.dimensions() != property_dimensions {
                    return Err(Error::InvalidParameter(format!(
                        "all blocks must have the same property dimensions, got [{}] and [{}]",
                        block.properties.dimensions().join(", "),
                        property_dimensions.join(", "),
                    )));
                }

                // check gradients are the same as those of the first block
                if GradientMap::new(block) != gradient_map {
                    return Err(Error::InvalidParameter(
                        "all blocks must have the same set of gradients, with \
                        the same sample, property and component dimensions, \
                        and the same must be true for gradients of gradients".into(),
                    ));
                }

            }
        }

        Ok(TensorMap {
            keys: keys,
            blocks,
            info: BTreeMap::new(),
            info_keys: Vec::new(),
        })
    }

    /// Try to copy this `TensorMap`. This can fail if we are unable to copy the
    /// underlying `mts_array_t` data array
    pub fn try_clone(&self) -> Result<TensorMap, Error> {
        let mut blocks = Vec::new();
        for block in &self.blocks {
            blocks.push(block.try_clone()?);
        }

        return Ok(TensorMap {
            keys: Arc::clone(&self.keys),
            blocks,
            info: self.info.clone(),
            info_keys: self.info_keys.clone()
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

    /// Move the all the given dimensions from the components to the properties
    /// for each block in this `TensorMap`.
    pub fn components_to_properties(&self, dimensions: &[&str]) -> Result<TensorMap, Error> {
        let mut clone = self.try_clone()?;

        if dimensions.is_empty() {
            return Ok(clone);
        }

        for block in &mut clone.blocks {
            for dimension in dimensions {
                block.components_to_properties(dimension)?;
            }
        }

        return Ok(clone);
    }

    /// Get the info map associated with this `TensorMap`.
    pub fn info(&self) -> &BTreeMap<String, ConstCString> {
        &self.info
    }

    /// Get the list of info keys as C-compatible strings
    pub fn info_keys_c(&self) -> &[ConstCString] {
        &self.info_keys
    }

    /// Get the info value associated with `key` for this `TensorMap`.
    pub fn get_info(&self, key: &str) -> Option<&ConstCString> {
        self.info.get(key)
    }

    /// Set the info value associated with `key` for this `TensorMap`.
    pub fn add_info(&mut self, key: &str, value: ConstCString) {
        if !self.info.contains_key(key) {
            self.info_keys.push(
                ConstCString::new(
                    CString::new(key.to_owned()).expect("invalid C string")
                )
            );
        }
        self.info.insert(key.to_owned(), value);
    }
}


#[cfg(test)]
mod tests {
    use crate::data::TestArray;

    use super::*;
    use super::utils::example_labels;

    #[test]
    #[allow(clippy::too_many_lines)]
    fn blocks_validation() {
        let block_1 = TensorBlock::new(
            TestArray::new(vec![1, 1, 1]),
            example_labels(&["samples"], &[0]),
            vec![example_labels(&["components"], &[0])],
            example_labels(&["properties"], &[0]),
        ).unwrap();

        let block_2 = TensorBlock::new(
            TestArray::new(vec![2, 3, 1]),
            example_labels(&["samples"], &[0, 1]),
            vec![example_labels(&["components"], &[0, 1, 2])],
            example_labels(&["properties"], &[0]),
        ).unwrap();

        let result = TensorMap::new(
            example_labels(&["keys"], &[0, 1]),
            vec![block_1, block_2],
        );
        assert!(result.is_ok());
        // also check we have an empty info block
        let mut result = result.unwrap();
        assert!(result.info.is_empty());
        // also check we can set info
        result.info.insert("key".to_string(),
                          ConstCString::new(CString::new("value").expect("CString::new failed")));

        /**********************************************************************/
        let block_1 = TensorBlock::new(
            TestArray::new(vec![1, 1]),
            example_labels(&["samples"], &[0]),
            vec![],
            example_labels(&["properties"], &[0]),
        ).unwrap();

        let block_2 = TensorBlock::new(
            TestArray::new(vec![2, 1]),
            example_labels(&["something_else"], &[0, 1]),
            vec![],
            example_labels(&["properties"], &[0]),
        ).unwrap();

        let result = TensorMap::new(
            example_labels(&["keys"], &[0, 1]),
            vec![block_1, block_2],
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: all blocks must have the same sample dimensions, \
            got [something_else] and [samples]"
        );

        /**********************************************************************/
        let block_1 = TensorBlock::new(
            TestArray::new(vec![1, 1, 1]),
            example_labels(&["samples"], &[0]),
            vec![example_labels(&["components"], &[0])],
            example_labels(&["properties"], &[0]),
        ).unwrap();

        let block_2 = TensorBlock::new(
            TestArray::new(vec![2, 1]),
            example_labels(&["samples"], &[0, 1]),
            vec![],
            example_labels(&["properties"], &[0]),
        ).unwrap();

        let result = TensorMap::new(
            example_labels(&["keys"], &[0, 1]),
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
            example_labels(&["samples"], &[0]),
            vec![example_labels(&["components"], &[0])],
            example_labels(&["properties"], &[0]),
        ).unwrap();

        let block_2 = TensorBlock::new(
            TestArray::new(vec![2, 3, 1]),
            example_labels(&["samples"], &[0, 1]),
            vec![example_labels(&["something_else"], &[0, 1, 2])],
            example_labels(&["properties"], &[0]),
        ).unwrap();

        let result = TensorMap::new(
            example_labels(&["keys"], &[0, 1]),
            vec![block_1, block_2],
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: all blocks must have the same component dimensions, \
            got [something_else] and [components]"
        );

        /**********************************************************************/
        let block_1 = TensorBlock::new(
            TestArray::new(vec![1, 1]),
            example_labels(&["samples"], &[0]),
            vec![],
            example_labels(&["properties"], &[0]),
        ).unwrap();

        let block_2 = TensorBlock::new(
            TestArray::new(vec![2, 1]),
            example_labels(&["samples"], &[0, 1]),
            vec![],
            example_labels(&["something_else"], &[0]),
        ).unwrap();

        let result = TensorMap::new(
            example_labels(&["keys"], &[0, 1]),
            vec![block_1, block_2],
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: all blocks must have the same property dimensions, \
            got [something_else] and [properties]"
        );

        // TODO: check error messages for gradients
    }

    #[test]
    fn blocks_device_mismatch() {
        let block_1 = TensorBlock::new(
            TestArray::new(vec![1, 1]),
            example_labels(&["samples"], &[0]),
            vec![],
            example_labels(&["properties"], &[0]),
        ).unwrap();

        let block_2 = TensorBlock::new(
            TestArray::new_other_device(vec![2, 1]),
            example_labels(&["samples"], &[0, 1]),
            vec![],
            example_labels(&["properties"], &[0]),
        ).unwrap();

        let result = TensorMap::new(
            example_labels(&["keys"], &[0, 1]),
            vec![block_1, block_2],
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: tried to build a TensorMap from blocks on \
            different devices: at least 'CPU:0' and 'CUDA:0' were detected"
        );
    }

    #[test]
    fn blocks_dtype_mismatch() {
        let block_1 = TensorBlock::new(
            TestArray::new(vec![1, 1]),
            example_labels(&["samples"], &[0]),
            vec![],
            example_labels(&["properties"], &[0]),
        ).unwrap();

        let block_2 = TensorBlock::new(
            TestArray::new_other_dtype(vec![2, 1]),
            example_labels(&["samples"], &[0, 1]),
            vec![],
            example_labels(&["properties"], &[0]),
        ).unwrap();

        let result = TensorMap::new(
            example_labels(&["keys"], &[0, 1]),
            vec![block_1, block_2],
        );
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: tried to build a TensorMap from blocks with \
            different dtypes: at least 'f64' and 'f32' were detected"
        );
    }
}
