use std::sync::Arc;
use std::collections::HashMap;

use crate::{Labels, Error, aml_array_t, get_data_origin};

/// Basic building block for descriptor. A single basic block contains a
/// 3-dimensional array, and three sets of labels (one for each dimension). The
/// sample labels are specific to this block, but components & feature labels
/// can be shared between blocks, or between values & gradients.
#[derive(Debug)]
pub struct BasicBlock {
    pub data: aml_array_t,
    pub(crate) samples: Labels,
    pub(crate) components: Arc<Labels>,
    pub(crate) features: Arc<Labels>,
}

fn check_data_label_shape(
    context: &str,
    data: &aml_array_t,
    samples: &Labels,
    components: &Labels,
    features: &Labels,
) -> Result<(), Error> {
    let (n_samples, n_components, n_features) = data.shape()?;
    if n_samples != samples.count() {
        return Err(Error::InvalidParameter(format!(
            "{}: the array shape along axis 0 is {} but we have {} sample labels",
            context, n_samples, samples.count()
        )));
    }

    if n_components != components.count() {
        return Err(Error::InvalidParameter(format!(
            "{}: the array shape along axis 1 is {} but we have {} components labels",
            context, n_components, components.count()
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
    /// Create a new `BasicBlock`, validating the shape of data & labels
    pub fn new(
        data: aml_array_t,
        samples: Labels,
        components: Arc<Labels>,
        features: Arc<Labels>,
    ) -> Result<BasicBlock, Error> {
        check_data_label_shape(
            "data and labels don't match", &data, &samples, &components, &features
        )?;

        return Ok(BasicBlock { data, samples, components, features });
    }

    /// Get the sample labels in this basic block
    pub fn samples(&self) -> &Labels {
        &self.samples
    }

    /// Get the components labels in this basic block
    pub fn components(&self) -> &Arc<Labels> {
        &self.components
    }

    /// Get the feature labels in this basic block
    pub fn features(&self) -> &Arc<Labels> {
        &self.features
    }
}

/// A single block in a descriptor, containing both values & optionally
/// gradients of these values w.r.t. any relevant quantity.
#[derive(Debug)]
pub struct Block {
    pub values: BasicBlock,
    gradients: HashMap<String, BasicBlock>,
}

impl Block {
    /// Create a new `Block` containing the given data, described by the
    /// `samples`, `components`, and `features` labels. The block is initialized
    /// without any gradients.
    pub fn new(
        data: aml_array_t,
        samples: Labels,
        components: Arc<Labels>,
        features: Arc<Labels>,
    ) -> Result<Block, Error> {
        Ok(Block {
            values: BasicBlock::new(data, samples, components, features)?,
            gradients: HashMap::new(),
        })
    }

    /// Check if this block contains gradients w.r.t. the `name` parameter
    pub fn has_gradient(&self, name: &str) -> bool {
        self.gradients.contains_key(name)
    }

    /// Get the list of gradients in this block.
    pub fn gradients_list(&self) -> Vec<&str> {
        self.gradients.keys().map(|s| &**s).collect()
    }

    /// Add a gradient to this block with the given name, samples and gradient
    /// array. The components and feature labels are assumed to match the ones of
    /// the values in this block.
    pub fn add_gradient(&mut self, name: &str, samples: Labels, gradient: aml_array_t) -> Result<(), Error> {
        if self.gradients.contains_key(name) {
            return Err(Error::InvalidParameter(format!(
                "gradient with respect to '{}' already exists for this block", name
            )))
        }

        if gradient.origin()? != self.values.data.origin()? {
            return Err(Error::InvalidParameter(format!(
                "the gradient array has a different origin ('{}') than the value array ('{}')",
                get_data_origin(gradient.origin()?),
                get_data_origin(self.values.data.origin()?),
            )))
        }

        // this is used as a special marker in the C API
        if name == "values" {
            return Err(Error::InvalidParameter(
                "can not store gradient with respect to 'values'".into()
            ))
        }

        if samples.size() < 1 || samples.names()[0] != "sample" {
            return Err(Error::InvalidParameter(
                "first variable in the gradients samples labels must be 'samples'".into()
            ))
        }

        let components = Arc::clone(self.values.components());
        let features = Arc::clone(self.values.features());
        check_data_label_shape(
            "gradient data and labels don't match", &gradient, &samples, &components, &features
        )?;

        self.gradients.insert(name.into(), BasicBlock {
            data: gradient,
            samples,
            components,
            features
        });

        return Ok(())
    }

    /// Get the gradients w.r.t. `name` in this block or None.
    pub fn get_gradient(&self, name: &str) -> Option<&BasicBlock> {
        self.gradients.get(name)
    }
}

#[cfg(test)]
mod tests {
    use crate::{LabelValue, LabelsBuilder};
    use crate::data::TestArray;

    use super::*;

    #[test]
    fn gradients() {
        let mut samples = LabelsBuilder::new(vec!["a", "b"]);
        samples.add(vec![LabelValue::new(0), LabelValue::new(0)]);
        samples.add(vec![LabelValue::new(0), LabelValue::new(1)]);
        samples.add(vec![LabelValue::new(0), LabelValue::new(2)]);
        samples.add(vec![LabelValue::new(3), LabelValue::new(2)]);

        let mut components = LabelsBuilder::new(vec!["c", "d"]);
        components.add(vec![LabelValue::new(-1), LabelValue::new(-4)]);
        components.add(vec![LabelValue::new(-2), LabelValue::new(-5)]);
        components.add(vec![LabelValue::new(-3), LabelValue::new(-6)]);
        let components = Arc::new(components.finish());

        let mut features = LabelsBuilder::new(vec!["f"]);
        features.add(vec![LabelValue::new(0)]);
        features.add(vec![LabelValue::new(1)]);
        features.add(vec![LabelValue::new(2)]);
        features.add(vec![LabelValue::new(3)]);
        features.add(vec![LabelValue::new(4)]);
        features.add(vec![LabelValue::new(5)]);
        features.add(vec![LabelValue::new(6)]);
        let features = Arc::new(features.finish());

        let data = aml_array_t::new(Box::new(TestArray::new((4, 3, 7))));

        let mut block = Block::new(data, samples.finish(), components, features).unwrap();
        assert!(block.gradients_list().is_empty());

        let gradient = aml_array_t::new(Box::new(TestArray::new((3, 3, 7))));
        let mut gradient_samples = LabelsBuilder::new(vec!["sample", "bar"]);
        gradient_samples.add(vec![LabelValue::new(0), LabelValue::new(0)]);
        gradient_samples.add(vec![LabelValue::new(1), LabelValue::new(1)]);
        gradient_samples.add(vec![LabelValue::new(3), LabelValue::new(-2)]);

        block.add_gradient("foo", gradient_samples.finish(), gradient).unwrap();

        assert_eq!(block.gradients_list(), ["foo"]);
        assert!(block.has_gradient("foo"));

        assert!(block.get_gradient("bar").is_none());
        let basic_block = block.get_gradient("foo").unwrap();

        assert_eq!(basic_block.samples().names(), ["sample", "bar"]);
        assert_eq!(basic_block.components().names(), ["c", "d"]);
        assert_eq!(basic_block.features().names(), ["f"]);
    }
}
