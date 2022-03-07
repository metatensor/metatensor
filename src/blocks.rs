use std::sync::Arc;
use std::collections::HashMap;

use crate::{Labels, Error, aml_array_t, get_data_origin};

/// Basic building block for descriptor. A single basic block contains a 3
/// array, and three sets of labels (one for each dimension). The sample labels
/// are specific to this block, but symmetric & feature labels can be shared
/// between blocks, or between values & gradients.
pub struct BasicBlock {
    pub data: aml_array_t,
    pub(crate) samples: Labels,
    pub(crate) symmetric: Arc<Labels>,
    pub(crate) features: Arc<Labels>,
}

fn check_data_label_shape(
    context: &str,
    data: &aml_array_t,
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
    /// Create a new `BasicBlock`, validating the shape of data & labels
    pub fn new(
        data: aml_array_t,
        samples: Labels,
        symmetric: Arc<Labels>,
        features: Arc<Labels>,
    ) -> Result<BasicBlock, Error> {
        check_data_label_shape(
            "data and labels don't match", &data, &samples, &symmetric, &features
        )?;

        return Ok(BasicBlock { data, samples, symmetric, features });
    }

    /// Get the sample labels in this basic block
    pub fn samples(&self) -> &Labels {
        &self.samples
    }

    /// Get the symmetric labels in this basic block
    pub fn symmetric(&self) -> &Arc<Labels> {
        &self.symmetric
    }

    /// Get the feature labels in this basic block
    pub fn features(&self) -> &Arc<Labels> {
        &self.features
    }
}

/// A single block in a descriptor, containing both values & optionally
/// gradients of these values w.r.t. any relevant quantity.
pub struct Block {
    pub values: BasicBlock,
    gradients: HashMap<String, BasicBlock>,
}

impl Block {
    /// Create a new `Block` containing the given data, described by the
    /// `samples`, `symmetric`, and `features` labels. The block is initialized
    /// without any gradients.
    pub fn new(
        data: aml_array_t,
        samples: Labels,
        symmetric: Arc<Labels>,
        features: Arc<Labels>,
    ) -> Result<Block, Error> {
        Ok(Block {
            values: BasicBlock::new(data, samples, symmetric, features)?,
            gradients: HashMap::new(),
        })
    }

    /// Check if this block contains gradients w.r.t. the `name` parameter
    pub fn has_gradients(&self, name: &str) -> bool {
        self.gradients.contains_key(name)
    }

    /// Get the list of gradients in this block.
    pub fn gradients_list(&self) -> Vec<&str> {
        self.gradients.keys().map(|s| &**s).collect()
    }

    /// Add a gradient to this block with the given name, samples and gradient
    /// array. The symmetric and feature labels are assumed to match the ones of
    /// the values in this block.
    pub fn add_gradient(&mut self, name: &str, samples: Labels, gradient: aml_array_t) -> Result<(), Error> {
        if self.gradients.contains_key(name) {
            return Err(Error::InvalidParameter(format!(
                "gradient with respect to '{}' already exists for this block", name
            )))
        }

        if gradient.origin() != self.values.data.origin() {
            return Err(Error::InvalidParameter(format!(
                "the gradient array has a different origin ('{}') than the value array ('{}')",
                get_data_origin(gradient.origin()),
                get_data_origin(self.values.data.origin()),
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

    /// Get the gradients w.r.t. `name` in this block or None.
    pub fn get_gradient(&self, name: &str) -> Option<&BasicBlock> {
        self.gradients.get(name)
    }
}

#[cfg(test)]
mod tests {
    use once_cell::sync::Lazy;
    use crate::data::{DataStorage, register_data_origin, DataOrigin};
    use crate::{LabelValue, LabelsBuilder};

    use super::*;

    static DUMMY_DATA_ORIGIN: Lazy<DataOrigin> = Lazy::new(|| {
        register_data_origin("dummy test data".into())
    });

    struct DummyArray {
        shape: (usize, usize, usize),
    }

    impl DataStorage for DummyArray {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }

        fn origin(&self) -> crate::DataOrigin {
            *DUMMY_DATA_ORIGIN
        }

        fn create(&self, shape: (usize, usize, usize)) -> Box<dyn DataStorage> {
            Box::new(DummyArray { shape: shape })
        }

        fn shape(&self) -> (usize, usize, usize) {
            self.shape
        }

        fn reshape(&mut self, shape: (usize, usize, usize)) {
            self.shape = shape;
        }

        fn set_from(
            &mut self,
            _sample: usize,
            _features: std::ops::Range<usize>,
            _other: &dyn DataStorage,
            _sample_other: usize
        ) {
            unimplemented!()
        }
    }

    #[test]
    fn gradients() {
        let mut samples = LabelsBuilder::new(vec!["a", "b"]);
        samples.add(vec![LabelValue::from(0_i32), LabelValue::from(0_i32)]);
        samples.add(vec![LabelValue::from(0_i32), LabelValue::from(1_i32)]);
        samples.add(vec![LabelValue::from(0_i32), LabelValue::from(2_i32)]);
        samples.add(vec![LabelValue::from(3_i32), LabelValue::from(2_i32)]);

        let mut symmetric = LabelsBuilder::new(vec!["c", "d"]);
        symmetric.add(vec![LabelValue::from(-1_i32), LabelValue::from(-4_i32)]);
        symmetric.add(vec![LabelValue::from(-2_i32), LabelValue::from(-5_i32)]);
        symmetric.add(vec![LabelValue::from(-3_i32), LabelValue::from(-6_i32)]);
        let symmetric = Arc::new(symmetric.finish());

        let mut features = LabelsBuilder::new(vec!["f"]);
        features.add(vec![LabelValue::from(0_i32)]);
        features.add(vec![LabelValue::from(1_i32)]);
        features.add(vec![LabelValue::from(2_i32)]);
        features.add(vec![LabelValue::from(3_i32)]);
        features.add(vec![LabelValue::from(4_i32)]);
        features.add(vec![LabelValue::from(5_i32)]);
        features.add(vec![LabelValue::from(6_i32)]);
        let features = Arc::new(features.finish());

        let data = aml_array_t::new(Box::new(DummyArray { shape: (4, 3, 7) }));

        let mut block = Block::new(data, samples.finish(), symmetric, features).unwrap();
        assert!(block.gradients_list().is_empty());

        let gradient = aml_array_t::new(Box::new(DummyArray { shape: (3, 3, 7) }));
        let mut gradient_samples = LabelsBuilder::new(vec!["sample", "bar"]);
        gradient_samples.add(vec![LabelValue::from(0_i32), LabelValue::from(0_i32)]);
        gradient_samples.add(vec![LabelValue::from(1_i32), LabelValue::from(1_i32)]);
        gradient_samples.add(vec![LabelValue::from(3_i32), LabelValue::from(-2_i32)]);

        block.add_gradient("foo", gradient_samples.finish(), gradient).unwrap();

        assert_eq!(block.gradients_list(), ["foo"]);
        assert!(block.has_gradients("foo"));

        assert!(block.get_gradient("bar").is_none());
        let basic_block = block.get_gradient("foo").unwrap();

        assert_eq!(basic_block.samples().names(), ["sample", "bar"]);
        assert_eq!(basic_block.symmetric().names(), ["c", "d"]);
        assert_eq!(basic_block.features().names(), ["f"]);
    }
}
