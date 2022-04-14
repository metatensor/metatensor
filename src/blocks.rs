use std::sync::Arc;
use std::ffi::CString;
use std::collections::{HashMap, BTreeSet};

use crate::utils::ConstCString;
use crate::{Labels, LabelsBuilder};
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
    pub(crate) components: Vec<Arc<Labels>>,
    pub(crate) features: Arc<Labels>,
}

fn check_data_and_labels(
    context: &str,
    data: &aml_array_t,
    samples: &Labels,
    components: &[Arc<Labels>],
    features: &Labels,
) -> Result<(), Error> {
    let shape = data.shape()?;

    if shape.len() != components.len() + 2 {
        return Err(Error::InvalidParameter(format!(
            "{}: the array has {} dimensions, but we have {} separate labels",
            context, shape.len(), components.len() + 2
        )));
    }

    if shape[0] != samples.count() {
        return Err(Error::InvalidParameter(format!(
            "{}: the array shape along axis 0 is {} but we have {} sample labels",
            context, shape[0], samples.count()
        )));
    }

    // ensure that all component labels have different names
    let n_components = components.iter().map(|c| c.names()).collect::<BTreeSet<_>>().len();
    if n_components != components.len() {
        return Err(Error::InvalidParameter(format!(
            "{}: some of the component names appear more than once in component labels",
            context,
        )));
    }

    let mut dimension = 1;
    for component in components {
        if shape[dimension] != component.count() {
            return Err(Error::InvalidParameter(format!(
                "{}: the array shape along axis {} is {} but we have {} entries \
                for the corresponding component",
                context, dimension, shape[dimension], component.count(),
            )));
        }
        dimension += 1;
    }

    if shape[dimension] != features.count() {
        return Err(Error::InvalidParameter(format!(
            "{}: the array shape along axis {} is {} but we have {} features labels",
            context, dimension, shape[dimension], features.count()
        )));
    }

    Ok(())
}

fn check_component_labels(components: &[Arc<Labels>]) -> Result<(), Error> {
    for (i, component) in components.iter().enumerate() {
        if component.size() != 1 {
            return Err(Error::InvalidParameter(format!(
                "component labels must have a single variable, got {}: [{}] for component {}",
                component.size(), component.names().join(", "), i
            )));
        }
    }
    Ok(())
}

impl BasicBlock {
    /// Create a new `BasicBlock`, validating the shape of data & labels
    pub fn new(
        data: aml_array_t,
        samples: Labels,
        components: Vec<Arc<Labels>>,
        features: Arc<Labels>,
    ) -> Result<BasicBlock, Error> {
        check_data_and_labels(
            "data and labels don't match", &data, &samples, &components, &features
        )?;

        check_component_labels(&components)?;

        return Ok(BasicBlock { data, samples, components, features });
    }

    /// Get the sample labels in this basic block
    pub fn samples(&self) -> &Labels {
        &self.samples
    }

    /// Get the components labels in this basic block
    pub fn components(&self) -> &[Arc<Labels>] {
        &self.components
    }

    /// Get the feature labels in this basic block
    pub fn features(&self) -> &Arc<Labels> {
        &self.features
    }

    fn components_to_features(&mut self, variables: &[&str]) -> Result<(), Error> {
        debug_assert!(!variables.is_empty());

        let mut component_axis = None;
        for (component_i, component) in self.components.iter().enumerate() {
            if component.names() == variables {
                component_axis = Some(component_i);
                break;
            }
        }

        let component_axis = component_axis.ok_or_else(|| Error::InvalidParameter(format!(
            "unable to find [{}] in the components ", variables.join(", ")
        )))?;

        let moved_component = self.components.remove(component_axis);

        // construct the new feature with old features and the components
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

        let mut new_shape = self.data.shape()?.to_vec();
        let features_axis = new_shape.len() - 1;
        new_shape[features_axis] = new_features.count();
        new_shape.remove(component_axis + 1);

        self.data.swap_axes(component_axis + 1, features_axis - 1)?;
        self.data.reshape(&new_shape)?;

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
        components: Vec<Arc<Labels>>,
        features: Arc<Labels>,
    ) -> Result<Block, Error> {
        Ok(Block {
            values: BasicBlock::new(data, samples, components, features)?,
            gradients: HashMap::new(),
            gradient_parameters: Vec::new(),
        })
    }

    /// Get all gradients defined in this block
    pub fn gradients(&self) -> &HashMap<String, BasicBlock> {
        &self.gradients
    }

    /// Get the list of gradients in this block for the C API
    pub fn gradient_parameters_c(&self) -> &[ConstCString] {
        &self.gradient_parameters
    }

    /// Add a gradient with respect to `parameter` to this block.
    ///
    /// The gradient `data` is given as an array, and the samples and components
    /// labels must be provided. The feature labels are assumed to match the
    /// ones of the values in this block.
    ///
    /// The components labels must contain at least the same entries as the
    /// value components labels, and can prepend other components labels.
    pub fn add_gradient(
        &mut self,
        parameter: &str,
        data: aml_array_t,
        samples: Labels,
        components: Vec<Arc<Labels>>,
    ) -> Result<(), Error> {
        if self.gradients.contains_key(parameter) {
            return Err(Error::InvalidParameter(format!(
                "gradient with respect to '{}' already exists for this block", parameter
            )))
        }

        if data.origin()? != self.values.data.origin()? {
            return Err(Error::InvalidParameter(format!(
                "the gradient array has a different origin ('{}') than the value array ('{}')",
                get_data_origin(data.origin()?),
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

        check_component_labels(&components)?;
        if self.values.components().len() > components.len() {
            return Err(Error::InvalidParameter(
                "gradients components should contain at least as many labels \
                as the values components".into()
            ))
        }
        let extra_gradient_components = components.len() - self.values.components().len();
        for (component_i, (gradient_labels, values_labels)) in components.iter()
            .skip(extra_gradient_components)
            .zip(self.values.components())
            .enumerate() {
                if gradient_labels != values_labels {
                    return Err(Error::InvalidParameter(format!(
                        "gradients and values components mismatch for values \
                        component {} (the corresponding names are [{}])",
                        component_i, values_labels.names().join(", ")
                    )))
                }
            }

        let features = Arc::clone(self.values.features());
        check_data_and_labels(
            "gradient data and labels don't match", &data, &samples, &components, &features
        )?;

        self.gradients.insert(parameter.into(), BasicBlock {
            data,
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

        self.values.components_to_features(variables)?;
        for gradient in self.gradients.values_mut() {
            gradient.components_to_features(variables)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{LabelValue, LabelsBuilder};
    use crate::data::TestArray;

    use super::*;

    fn example_labels(name: &str, count: usize) -> Labels {
        let mut labels = LabelsBuilder::new(vec![name]);
        for i in 0..count {
            labels.add(vec![LabelValue::from(i)]);
        }
        return labels.finish();
    }

    #[test]
    fn no_components() {
        let samples = example_labels("samples", 4);
        let features = Arc::new(example_labels("features", 7));
        let data = aml_array_t::new(Box::new(TestArray::new(vec![4, 7])));
        let result = Block::new(data, samples.clone(), Vec::new(), features.clone());
        assert!(result.is_ok());

        let data = aml_array_t::new(Box::new(TestArray::new(vec![3, 7])));
        let result = Block::new(data, samples.clone(), Vec::new(), features.clone());
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: data and labels don't match: the array shape \
            along axis 0 is 3 but we have 4 sample labels"
        );

        let data = aml_array_t::new(Box::new(TestArray::new(vec![4, 9])));
        let result = Block::new(data, samples.clone(), Vec::new(), features.clone());
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: data and labels don't match: the array shape \
            along axis 1 is 9 but we have 7 features labels"
        );

        let data = aml_array_t::new(Box::new(TestArray::new(vec![4, 1, 7])));
        let result = Block::new(data, samples, Vec::new(), features);
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: data and labels don't match: the array has \
            3 dimensions, but we have 2 separate labels"
        );
    }

    #[test]
    fn multiple_components() {
        let component_1 = Arc::new(example_labels("component_1", 4));
        let component_2 = Arc::new(example_labels("component_2", 3));

        let samples = example_labels("samples", 3);
        let features = Arc::new(example_labels("features", 2));
        let data = aml_array_t::new(Box::new(TestArray::new(vec![3, 4, 2])));
        let components = vec![Arc::clone(&component_1)];
        let result = Block::new(data, samples.clone(), components, features.clone());
        assert!(result.is_ok());

        let data = aml_array_t::new(Box::new(TestArray::new(vec![3, 4, 3, 2])));
        let components = vec![Arc::clone(&component_1), Arc::clone(&component_2)];
        let result = Block::new(data, samples.clone(), components, features.clone());
        assert!(result.is_ok());

        let data = aml_array_t::new(Box::new(TestArray::new(vec![3, 4, 2])));
        let components = vec![Arc::clone(&component_1), Arc::clone(&component_2)];
        let result = Block::new(data, samples.clone(), components, features.clone());
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: data and labels don't match: the array has 3 \
            dimensions, but we have 4 separate labels"
        );

        let data = aml_array_t::new(Box::new(TestArray::new(vec![3, 4, 4, 2])));
        let components = vec![Arc::clone(&component_1), Arc::clone(&component_2)];
        let result = Block::new(data, samples.clone(), components, features.clone());
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: data and labels don't match: the array shape \
            along axis 2 is 4 but we have 3 entries for the corresponding component"
        );

        let data = aml_array_t::new(Box::new(TestArray::new(vec![3, 4, 4, 2])));
        let components = vec![Arc::clone(&component_1), Arc::clone(&component_1)];
        let result = Block::new(data, samples.clone(), components, features.clone());
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: data and labels don't match: some of the \
            component names appear more than once in component labels"
        );

        let data = aml_array_t::new(Box::new(TestArray::new(vec![3, 1, 2])));
        let mut components = LabelsBuilder::new(vec!["component_1", "component_2"]);
        components.add(vec![LabelValue::from(0), LabelValue::from(1)]);

        let result = Block::new(data, samples, vec![Arc::new(components.finish())], features);
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: component labels must have a single variable, \
            got 2: [component_1, component_2] for component 0"
        );
    }

    mod gradients {
        use super::*;

        #[test]
        fn values_without_components() {
            let samples = example_labels("samples", 4);
            let features = Arc::new(example_labels("features", 7));
            let data = aml_array_t::new(Box::new(TestArray::new(vec![4, 7])));
            let mut block = Block::new(data, samples, vec![], features).unwrap();
            assert!(block.gradients().is_empty());

            let gradient = aml_array_t::new(Box::new(TestArray::new(vec![3, 7])));
            let mut gradient_samples = LabelsBuilder::new(vec!["sample", "foo"]);
            gradient_samples.add(vec![LabelValue::new(0), LabelValue::new(0)]);
            gradient_samples.add(vec![LabelValue::new(1), LabelValue::new(1)]);
            gradient_samples.add(vec![LabelValue::new(3), LabelValue::new(-2)]);
            block.add_gradient("foo", gradient, gradient_samples.finish(), vec![]).unwrap();

            let gradient = aml_array_t::new(Box::new(TestArray::new(vec![3, 5, 7])));
            let gradient_samples = example_labels("sample", 3);
            let component = Arc::new(example_labels("component", 5));
            block.add_gradient("component", gradient, gradient_samples, vec![component]).unwrap();

            let mut gradients_list = block.gradients().keys().collect::<Vec<_>>();
            gradients_list.sort_unstable();
            assert_eq!(gradients_list, ["component", "foo"]);

            let basic_block = block.gradients().get("foo").unwrap();
            assert_eq!(basic_block.samples().names(), ["sample", "foo"]);
            assert!(basic_block.components().is_empty());
            assert_eq!(basic_block.features().names(), ["features"]);

            let basic_block = block.gradients().get("component").unwrap();
            assert_eq!(basic_block.samples().names(), ["sample"]);
            assert_eq!(basic_block.components().len(), 1);
            assert_eq!(basic_block.components()[0].names(), ["component"]);
            assert_eq!(basic_block.features().names(), ["features"]);

            assert!(block.gradients().get("baz").is_none());
        }

        #[test]
        fn values_with_components() {
            let samples = example_labels("samples", 4);
            let component = Arc::new(example_labels("component", 5));
            let features = Arc::new(example_labels("features", 7));
            let data = aml_array_t::new(Box::new(TestArray::new(vec![4, 5, 7])));
            let mut block = Block::new(data, samples, vec![component.clone()], features).unwrap();

            let gradient = aml_array_t::new(Box::new(TestArray::new(vec![3, 5, 7])));
            let gradient_samples = example_labels("sample", 3);
            let result = block.add_gradient("basic", gradient, gradient_samples.clone(), vec![component.clone()]);
            assert!(result.is_ok());

            let gradient = aml_array_t::new(Box::new(TestArray::new(vec![3, 3, 5, 7])));
            let component_2 = Arc::new(example_labels("component_2", 3));
            let components = vec![component_2.clone(), component.clone()];
            let result = block.add_gradient("components", gradient, gradient_samples.clone(), components);
            assert!(result.is_ok());

            let gradient = aml_array_t::new(Box::new(TestArray::new(vec![3, 3, 5, 7])));
            let components = vec![component, component_2];
            let result = block.add_gradient("wrong", gradient, gradient_samples, components);
            assert_eq!(
                result.unwrap_err().to_string(),
                "invalid parameter: gradients and values components mismatch \
                for values component 0 (the corresponding names are [component])"
            );
        }
    }

    #[cfg(feature = "ndarray")]
    mod components_to_features {
        use super::*;
        use ndarray::ArrayD;

        #[test]
        fn one_component() {
            let mut block = Block::new(
                aml_array_t::new(Box::new(ArrayD::from_elem(vec![3, 2, 3], 1.0))),
                example_labels("samples", 3),
                vec![Arc::new(example_labels("components", 2))],
                Arc::new(example_labels("features", 3)),
            ).unwrap();

            let mut grad_samples = LabelsBuilder::new(vec!["sample", "parameter"]);
            grad_samples.add(vec![LabelValue::new(0), LabelValue::new(2)]);
            grad_samples.add(vec![LabelValue::new(1), LabelValue::new(2)]);
            let grad_samples = grad_samples.finish();

            block.add_gradient(
                "parameter",
                aml_array_t::new(Box::new(ArrayD::from_elem(vec![2, 2, 3], 11.0))),
                grad_samples,
                vec![Arc::new(example_labels("components", 2))],
            ).unwrap();

            /******************************************************************/

            block.components_to_features(&["components"]).unwrap();

            assert_eq!(block.values.samples().names(), ["samples"]);
            assert_eq!(block.values.samples().count(), 3);
            assert_eq!(block.values.samples()[0], [LabelValue::new(0)]);
            assert_eq!(block.values.samples()[1], [LabelValue::new(1)]);
            assert_eq!(block.values.samples()[2], [LabelValue::new(2)]);

            assert_eq!(block.values.components().len(), 0);

            assert_eq!(block.values.features().names(), ["components", "features"]);
            assert_eq!(block.values.features().count(), 6);
            assert_eq!(block.values.features()[0], [LabelValue::new(0), LabelValue::new(0)]);
            assert_eq!(block.values.features()[1], [LabelValue::new(0), LabelValue::new(1)]);
            assert_eq!(block.values.features()[2], [LabelValue::new(0), LabelValue::new(2)]);
            assert_eq!(block.values.features()[3], [LabelValue::new(1), LabelValue::new(0)]);
            assert_eq!(block.values.features()[4], [LabelValue::new(1), LabelValue::new(1)]);
            assert_eq!(block.values.features()[5], [LabelValue::new(1), LabelValue::new(2)]);

            assert_eq!(block.values.data.as_array(), ArrayD::from_elem(vec![3, 6], 1.0));

            let gradient = block.get_gradient("parameter").unwrap();
            assert_eq!(gradient.samples().names(), ["sample", "parameter"]);
            assert_eq!(gradient.samples().count(), 2);
            assert_eq!(gradient.samples()[0], [LabelValue::new(0), LabelValue::new(2)]);
            assert_eq!(gradient.samples()[1], [LabelValue::new(1), LabelValue::new(2)]);

            assert_eq!(gradient.data.as_array(), ArrayD::from_elem(vec![2, 6], 11.0));
        }

        #[test]
        fn multiple_components() {
            let data = ArrayD::from_shape_vec(vec![2, 2, 3, 2], vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0,
                -1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.0, 4.0, -5.0, 5.0, -6.0, 6.0,
            ]).unwrap();

            let components = vec![
                Arc::new(example_labels("component_1", 2)),
                Arc::new(example_labels("component_2", 3)),
            ];
            let mut block = Block::new(
                aml_array_t::new(Box::new(data)),
                example_labels("samples", 2),
                components.clone(),
                Arc::new(example_labels("features", 2)),
            ).unwrap();

            let mut grad_samples = LabelsBuilder::new(vec!["sample", "parameter"]);
            grad_samples.add(vec![LabelValue::new(0), LabelValue::new(2)]);
            grad_samples.add(vec![LabelValue::new(0), LabelValue::new(3)]);
            grad_samples.add(vec![LabelValue::new(1), LabelValue::new(2)]);
            let grad_samples = grad_samples.finish();

            block.add_gradient(
                "parameter",
                aml_array_t::new(Box::new(ArrayD::from_elem(vec![3, 2, 3, 2], 11.0))),
                grad_samples,
                components
            ).unwrap();

            /******************************************************************/

            block.components_to_features(&["component_1"]).unwrap();

            assert_eq!(block.values.samples().names(), ["samples"]);
            assert_eq!(block.values.samples().count(), 2);
            assert_eq!(block.values.samples()[0], [LabelValue::new(0)]);
            assert_eq!(block.values.samples()[1], [LabelValue::new(1)]);

            assert_eq!(block.values.components().len(), 1);
            assert_eq!(block.values.components()[0].names(), ["component_2"]);
            assert_eq!(block.values.components()[0].count(), 3);
            assert_eq!(block.values.components()[0][0], [LabelValue::new(0)]);
            assert_eq!(block.values.components()[0][1], [LabelValue::new(1)]);
            assert_eq!(block.values.components()[0][2], [LabelValue::new(2)]);

            assert_eq!(block.values.features().names(), ["component_1", "features"]);
            assert_eq!(block.values.features().count(), 4);
            assert_eq!(block.values.features()[0], [LabelValue::new(0), LabelValue::new(0)]);
            assert_eq!(block.values.features()[1], [LabelValue::new(0), LabelValue::new(1)]);
            assert_eq!(block.values.features()[2], [LabelValue::new(1), LabelValue::new(0)]);
            assert_eq!(block.values.features()[3], [LabelValue::new(1), LabelValue::new(1)]);

            let expected = ArrayD::from_shape_vec(vec![2, 3, 4], vec![
                1.0, 1.0, 4.0, 4.0, 2.0, 2.0, 5.0, 5.0, 3.0, 3.0, 6.0, 6.0,
                -1.0, 1.0, -4.0, 4.0, -2.0, 2.0, -5.0, 5.0, -3.0, 3.0, -6.0, 6.0,
            ]).unwrap();
            assert_eq!(block.values.data.as_array(), expected);

            let gradient = block.get_gradient("parameter").unwrap();
            assert_eq!(gradient.samples().names(), ["sample", "parameter"]);
            assert_eq!(gradient.samples().count(), 3);
            assert_eq!(gradient.samples()[0], [LabelValue::new(0), LabelValue::new(2)]);
            assert_eq!(gradient.samples()[1], [LabelValue::new(0), LabelValue::new(3)]);
            assert_eq!(gradient.samples()[2], [LabelValue::new(1), LabelValue::new(2)]);

            assert_eq!(gradient.data.as_array(), ArrayD::from_elem(vec![3, 3, 4], 11.0));
        }
    }
}
