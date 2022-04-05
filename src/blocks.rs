use std::sync::Arc;
use std::ffi::CString;
use std::collections::HashMap;

use crate::utils::ConstCString;
use crate::{Labels, LabelValue, LabelsBuilder};
use crate::{aml_array_t, get_data_origin};
use crate::Error;

/// Basic building block for descriptor. A single basic block contains a
/// 3-dimensional array, and three sets of labels (one for each dimension). The
/// sample labels are specific to this block, but components & feature labels
/// can be shared between blocks, or between values & gradients.
#[derive(Debug, Clone)]
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

    fn components_to_features(
        &mut self,
        variables: &[&str],
        new_components_to_component: &[usize],
        new_features_to_component: &[usize],
    ) -> Result<(), Error> {
        debug_assert!(!variables.is_empty());

        let (new_components, moved_component) = self.components.split(variables)?;

        // We want to be sure that the components are a full cartesian
        // product of `new_components` and `new_features`
        assert_eq!(new_components.count() * moved_component.count(), self.components.count());

        // construct the new feature with old features and moved_component
        let old_features = &self.features;
        let new_feature_names = moved_component.names().iter()
            .chain(old_features.names().iter())
            .copied()
            .collect();
        let mut new_features_builder = LabelsBuilder::new(new_feature_names);
        for new_feature in moved_component.iter() {
            for old_feature in old_features.iter() {
                let mut feature = new_feature.to_vec();
                feature.extend_from_slice(old_feature);
                new_features_builder.add(feature);
            }
        }
        let new_features = new_features_builder.finish();

        let new_shape = (
            self.samples.count(),
            new_components.count(),
            new_features.count()
        );
        let mut new_data = self.data.create(new_shape)?;

        let old_feature_count = self.features.count();
        let old_component_size = self.components.size();

        // move the data from the previous array to the new one. We can use
        // a double loop over moved_components and new_components since we
        // ensured above that the old component were a full cartesian
        // product of moved_components and new_components
        let mut old_component = vec![LabelValue::new(0); old_component_size];
        for (moved_component_i, moved_component) in moved_component.iter().enumerate() {
            let feature_start = moved_component_i * old_feature_count;
            let feature_stop = (moved_component_i + 1) * old_feature_count;

            for (&i, &value) in new_features_to_component.iter().zip(moved_component) {
                old_component[i] = value;
            }

            for (new_component_i, new_component) in new_components.iter().enumerate() {
                for (&i, &value) in new_components_to_component.iter().zip(new_component) {
                    old_component[i] = value;
                }

                let old_component_i = self.components.position(&old_component).expect("missing old component");
                new_data.move_component(
                    new_component_i,
                    feature_start..feature_stop,
                    &self.data,
                    old_component_i
                )?;
            }
        }

        self.data = new_data;
        // self.samples do not change
        self.components = Arc::new(new_components);
        self.features = Arc::new(new_features);

        Ok(())
    }
}

/// A single block in a descriptor, containing both values & optionally
/// gradients of these values w.r.t. any relevant quantity.
#[derive(Debug, Clone)]
pub struct Block {
    pub values: BasicBlock,
    gradients: HashMap<String, BasicBlock>,
    // all the keys from `self.gradients`, as C-compatible strings
    gradient_parameters: Vec<ConstCString>,
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
            gradient_parameters: Vec::new(),
        })
    }

    /// Check if this block contains gradients w.r.t. the `name` parameter
    pub fn has_gradient(&self, name: &str) -> bool {
        self.gradients.contains_key(name)
    }

    /// Get the list of gradients in this block.
    pub fn gradients_list(&self) -> Vec<&str> {
        self.gradient_parameters.iter().map(|s| s.as_str()).collect()
    }

    /// Get the list of gradients in this block for the C API
    pub fn gradients_list_c(&self) -> &[ConstCString] {
        &self.gradient_parameters
    }

    /// Add a gradient to this block with the given name, samples and gradient
    /// array. The components and feature labels are assumed to match the ones of
    /// the values in this block.
    pub fn add_gradient(&mut self, parameter: &str, samples: Labels, gradient: aml_array_t) -> Result<(), Error> {
        if self.gradients.contains_key(parameter) {
            return Err(Error::InvalidParameter(format!(
                "gradient with respect to '{}' already exists for this block", parameter
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
        if parameter == "values" {
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

        self.gradients.insert(parameter.into(), BasicBlock {
            data: gradient,
            samples,
            components,
            features
        });

        let parameter = ConstCString::new(CString::new(parameter.to_owned()).expect("invalid C string"));
        self.gradient_parameters.push(parameter);

        return Ok(())
    }

    /// Get the gradients w.r.t. `parameter` in this block or None.
    pub fn get_gradient(&self, parameter: &str) -> Option<&BasicBlock> {
        self.gradients.get(parameter)
    }

    pub(crate) fn components_to_features(&mut self, variables: &[&str]) -> Result<(), Error> {
        if variables.is_empty() {
            return Ok(());
        }

        // these two vector tell us how to organize the new component/features
        // to rebuild the previous component
        let mut new_components_to_component = Vec::new();
        let mut new_features_to_component = Vec::new();
        for (i, name) in self.values.components.names().iter().enumerate() {
            if variables.contains(name) {
                new_features_to_component.push(i);
            } else {
                new_components_to_component.push(i);
            }
        }

        self.values.components_to_features(
            variables,
            &new_components_to_component,
            &new_features_to_component,
        )?;

        for gradient in self.gradients.values_mut() {
            gradient.components_to_features(
                variables,
                &new_components_to_component,
                &new_features_to_component,
            )?;
        }

        Ok(())
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

    #[cfg(feature = "ndarray")]
    mod components_to_features {
        use super::*;
        use ndarray::{array, Array3};

        #[test]
        fn invariant() {
            let mut samples = LabelsBuilder::new(vec!["samples"]);
            samples.add(vec![LabelValue::new(0)]);
            samples.add(vec![LabelValue::new(1)]);
            let samples = samples.finish();

            let mut components = LabelsBuilder::new(vec!["components"]);
            components.add(vec![LabelValue::new(0)]);
            let components = components.finish();

            let mut features = LabelsBuilder::new(vec!["features"]);
            features.add(vec![LabelValue::new(0)]);
            features.add(vec![LabelValue::new(2)]);
            features.add(vec![LabelValue::new(-4)]);
            let features = features.finish();

            let mut block = Block::new(
                aml_array_t::new(Box::new(array![[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])),
                samples,
                Arc::new(components),
                Arc::new(features),
            ).unwrap();

            let mut grad_samples = LabelsBuilder::new(vec!["sample", "parameter"]);
            grad_samples.add(vec![LabelValue::new(0), LabelValue::new(2)]);
            grad_samples.add(vec![LabelValue::new(0), LabelValue::new(3)]);
            grad_samples.add(vec![LabelValue::new(1), LabelValue::new(2)]);
            let grad_samples = grad_samples.finish();

            block.add_gradient(
                "parameter",
                grad_samples,
                aml_array_t::new(Box::new(Array3::from_elem((3, 1, 3), 11.0)))
            ).unwrap();

            /******************************************************************/

            block.components_to_features(&["components"]).unwrap();

            assert_eq!(block.values.samples().names(), ["samples"]);
            assert_eq!(block.values.samples().count(), 2);
            assert_eq!(block.values.samples()[0], [LabelValue::new(0)]);
            assert_eq!(block.values.samples()[1], [LabelValue::new(1)]);

            assert_eq!(block.values.components().names(), ["_"]);
            assert_eq!(block.values.components().count(), 1);
            assert_eq!(block.values.components()[0], [LabelValue::new(0)]);

            assert_eq!(block.values.features().names(), ["components", "features"]);
            assert_eq!(block.values.features().count(), 3);
            assert_eq!(block.values.features()[0], [LabelValue::new(0), LabelValue::new(0)]);
            assert_eq!(block.values.features()[1], [LabelValue::new(0), LabelValue::new(2)]);
            assert_eq!(block.values.features()[2], [LabelValue::new(0), LabelValue::new(-4)]);

            assert_eq!(block.values.data.as_array(), array![
                [[1.0, 2.0, 3.0]],
                [[4.0, 5.0, 6.0]]
            ]);

            let gradient = block.get_gradient("parameter").unwrap();
            assert_eq!(gradient.samples().names(), ["sample", "parameter"]);
            assert_eq!(gradient.samples().count(), 3);
            assert_eq!(gradient.samples()[0], [LabelValue::new(0), LabelValue::new(2)]);
            assert_eq!(gradient.samples()[1], [LabelValue::new(0), LabelValue::new(3)]);
            assert_eq!(gradient.samples()[2], [LabelValue::new(1), LabelValue::new(2)]);

            assert_eq!(gradient.data.as_array(), Array3::from_elem((3, 1, 3), 11.0));
        }

        #[test]
        fn multiple_components() {
            let mut samples = LabelsBuilder::new(vec!["sample"]);
            samples.add(vec![LabelValue::new(0)]);
            samples.add(vec![LabelValue::new(1)]);
            let samples = samples.finish();

            let mut components = LabelsBuilder::new(vec!["component_1", "component_2"]);
            components.add(vec![LabelValue::new(-1), LabelValue::new(0)]);
            components.add(vec![LabelValue::new(-1), LabelValue::new(1)]);
            components.add(vec![LabelValue::new(-1), LabelValue::new(2)]);
            components.add(vec![LabelValue::new(0), LabelValue::new(0)]);
            components.add(vec![LabelValue::new(0), LabelValue::new(1)]);
            components.add(vec![LabelValue::new(0), LabelValue::new(2)]);
            components.add(vec![LabelValue::new(1), LabelValue::new(0)]);
            components.add(vec![LabelValue::new(1), LabelValue::new(1)]);
            components.add(vec![LabelValue::new(1), LabelValue::new(2)]);
            let components = components.finish();

            let mut features = LabelsBuilder::new(vec!["features"]);
            features.add(vec![LabelValue::new(0)]);
            let features = features.finish();

            let mut block = Block::new(
                aml_array_t::new(Box::new(array![
                    [[-1.0], [-2.0], [-3.0], [0.0], [0.0], [0.0], [1.0], [2.0], [3.0]],
                    [[-1.0], [-2.0], [-3.0], [0.0], [0.0], [0.0], [4.0], [5.0], [6.0]],
                ])),
                samples,
                Arc::new(components),
                Arc::new(features),
            ).unwrap();

            let mut grad_samples = LabelsBuilder::new(vec!["sample", "parameter"]);
            grad_samples.add(vec![LabelValue::new(0), LabelValue::new(2)]);
            grad_samples.add(vec![LabelValue::new(0), LabelValue::new(3)]);
            grad_samples.add(vec![LabelValue::new(1), LabelValue::new(2)]);
            let grad_samples = grad_samples.finish();

            block.add_gradient(
                "parameter",
                grad_samples,
                aml_array_t::new(Box::new(Array3::from_elem((3, 9, 1), 11.0)))
            ).unwrap();

            /******************************************************************/

            block.components_to_features(&["component_1"]).unwrap();

            assert_eq!(block.values.samples().names(), ["sample"]);
            assert_eq!(block.values.samples().count(), 2);
            assert_eq!(block.values.samples()[0], [LabelValue::new(0)]);
            assert_eq!(block.values.samples()[1], [LabelValue::new(1)]);

            assert_eq!(block.values.components().names(), ["component_2"]);
            assert_eq!(block.values.components().count(), 3);
            assert_eq!(block.values.components()[0], [LabelValue::new(0)]);
            assert_eq!(block.values.components()[1], [LabelValue::new(1)]);
            assert_eq!(block.values.components()[2], [LabelValue::new(2)]);

            assert_eq!(block.values.features().names(), ["component_1", "features"]);
            assert_eq!(block.values.features().count(), 3);
            assert_eq!(block.values.features()[0], [LabelValue::new(-1), LabelValue::new(0)]);
            assert_eq!(block.values.features()[1], [LabelValue::new(0), LabelValue::new(0)]);
            assert_eq!(block.values.features()[2], [LabelValue::new(1), LabelValue::new(0)]);

            assert_eq!(block.values.data.as_array(), array![
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-3.0, 0.0, 3.0]],
                [[-1.0, 0.0, 4.0], [-2.0, 0.0, 5.0], [-3.0, 0.0, 6.0]],
            ]);

            let gradient = block.get_gradient("parameter").unwrap();
            assert_eq!(gradient.samples().names(), ["sample", "parameter"]);
            assert_eq!(gradient.samples().count(), 3);
            assert_eq!(gradient.samples()[0], [LabelValue::new(0), LabelValue::new(2)]);
            assert_eq!(gradient.samples()[1], [LabelValue::new(0), LabelValue::new(3)]);
            assert_eq!(gradient.samples()[2], [LabelValue::new(1), LabelValue::new(2)]);

            assert_eq!(gradient.data.as_array(), Array3::from_elem((3, 3, 3), 11.0));
        }
    }
}
