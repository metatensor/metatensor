use std::sync::Arc;
use std::collections::HashMap;

use crate::{Labels, Error, aml_data_storage_t};

pub struct BasicBlock {
    pub data: aml_data_storage_t,
    pub(crate) samples: Labels,
    pub(crate) symmetric: Arc<Labels>,
    pub(crate) features: Arc<Labels>,
}

fn check_data_label_shape(
    context: &str,
    data: &aml_data_storage_t,
    samples: &Labels,
    symmetric: &Labels,
    features: &Labels,
) -> Result<(), Error> {
    let (n_samples, n_symmetric, n_features) = data.shape();
    if n_samples != samples.count() {
        return Err(Error::InvalidParameter(format!(
            "{}: the array shape along axis 0 is {} but we have {} sample labels",
            context, n_samples, samples.count()
        )));
    }

    if n_symmetric != symmetric.count() {
        return Err(Error::InvalidParameter(format!(
            "{}: the array shape along axis 1 is {} but we have {} symmetric labels",
            context, n_symmetric, symmetric.count()
        )));
    }

    if n_features != features.count() {
        return Err(Error::InvalidParameter(format!(
            "{}: the array shape along axis 2 is {} but we have {} features labels",
            context, n_features, features.count()
        )));
    }

    Ok(())
}

impl BasicBlock {
    pub fn new(
        data: aml_data_storage_t,
        samples: Labels,
        symmetric: Arc<Labels>,
        features: Arc<Labels>,
    ) -> Result<BasicBlock, Error> {
        check_data_label_shape(
            "data and labels don't match", &data, &samples, &symmetric, &features
        )?;

        return Ok(BasicBlock { data, samples, symmetric, features });
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
    ) -> Result<Block, Error> {
        Ok(Block {
            values: BasicBlock::new(data, samples, symmetric, features)?,
            gradients: HashMap::new(),
        })
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
                "first variable in the gradients samples labels must be 'samples'".into()
            ))
        }

        let symmetric = Arc::clone(self.values.symmetric());
        let features = Arc::clone(self.values.features());
        check_data_label_shape(
            "gradient data and labels don't match", &gradient, &samples, &symmetric, &features
        )?;

        self.gradients.insert(name.into(), BasicBlock {
            data: gradient,
            samples,
            symmetric,
            features
        });

        return Ok(())
    }

    pub fn get_gradient(&self, name: &str) -> Option<&BasicBlock> {
        self.gradients.get(name)
    }
}
