use std::sync::Arc;
use std::collections::HashMap;

use crate::{Labels, Error, aml_data_storage_t};

pub struct BasicBlock {
    pub data: aml_data_storage_t,
    pub(crate) samples: Labels,
    pub(crate) symmetric: Arc<Labels>,
    pub(crate) features: Arc<Labels>,
}

impl BasicBlock {
    pub fn new(
        data: aml_data_storage_t,
        samples: Labels,
        symmetric: Arc<Labels>,
        features: Arc<Labels>,
    ) -> BasicBlock {
        // TODO: checks on size
        return BasicBlock { data, samples, symmetric, features };
    }

    pub fn samples(&self) -> &Labels {
        &self.samples
    }

    pub fn symmetric(&self) -> &Arc<Labels> {
        &self.symmetric
    }

    pub fn features(&self) -> &Arc<Labels> {
        &self.features
    }
}

pub struct Block {
    pub values: BasicBlock,
    gradients: HashMap<String, BasicBlock>,
}

impl Block {
    pub fn new(
        data: aml_data_storage_t,
        samples: Labels,
        symmetric: Arc<Labels>,
        features: Arc<Labels>,
    ) -> Block {
        Block {
            values: BasicBlock::new(data, samples, symmetric, features),
            gradients: HashMap::new(),
        }
    }

    pub fn has_gradients(&self) -> bool {
        !self.gradients.is_empty()
    }

    pub fn add_gradient(&mut self, name: &str, samples: Labels, gradient: aml_data_storage_t) -> Result<(), Error> {
        if self.gradients.contains_key(name) {
            return Err(Error::InvalidParameter(format!(
                "gradient with respect to '{}' already exists for this block", name
            )))
        }

        // this is used as a special marker in the C API
        if name == "values" {
            return Err(Error::InvalidParameter(
                "can not store gradient with respect to 'values'".into()
            ))
        }

        if samples.names()[0] != "sample" {
            return Err(Error::InvalidParameter(
                "first variable in the samples Labels must be 'samples'".into()
            ))
        }

        // TODO: check shape for symmetric & features

        let gradient = BasicBlock::new(
            gradient,
            samples,
            Arc::clone(self.values.symmetric()),
            Arc::clone(self.values.features())
        );

        self.gradients.insert(name.into(), gradient);

        return Ok(())
    }

    pub fn get_gradient(&self, name: &str) -> Option<&BasicBlock> {
        self.gradients.get(name)
    }
}
