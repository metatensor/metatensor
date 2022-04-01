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

    fn components_to_features(&mut self, variables: &[&str]) -> Result<(), Error> {
        debug_assert!(!variables.is_empty());
        // these two vector tell us how to organize the new component/features
        // to rebuild the previous component
        let mut new_components_to_component = Vec::new();
        let mut new_features_to_component = Vec::new();
        for (i, name) in self.components.names().iter().enumerate() {
            if variables.contains(name) {
                new_features_to_component.push(i);
            } else {
                new_components_to_component.push(i);
            }
        }

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
#[derive(Debug)]
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

    /// Get the gradients already defined in this block
    pub fn gradients(&self) -> &HashMap<String, BasicBlock> {
        &self.gradients
    }

    /// Get the list of gradients in this block for the C API
    pub fn gradients_list_c(&self) -> &[ConstCString] {
        &self.gradient_parameters
    }

    /// Add gradient `data` with respect to `parameter` to this block.
    ///
    /// The gradient sample and component labels can be provided, while the
    /// feature labels are assumed to match the values. If the components are
    /// set to `None`, they are assumed to match the values'.
    pub fn add_gradient(
        &mut self,
        parameter: &str,
        data: aml_array_t,
        samples: Labels,
        components: impl Into<Option<Arc<Labels>>>,
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

        let components = components.into().unwrap_or_else(|| Arc::clone(self.values.components()));

        if **self.values.components() != Labels::single() {
            let values_components_names = self.values.components().names();
            if components.names().len() < values_components_names.len() {
                return Err(Error::InvalidParameter(format!(
                    "gradient component labels must contain at least the same \
                    number of variables as the value components, got {} but \
                    expected at least {}",
                    components.names().len(), values_components_names.len()
                )));
            }

            let new_variables = components.names().len() - values_components_names.len();
            if components.names()[new_variables..] != values_components_names {
                return Err(Error::InvalidParameter(format!(
                    "gradient component labels must end with the same variables \
                    as values component labels, got [{}] but the values contains [{}]",
                    components.names().join(", "), values_components_names.join(", ")
                )));
            }

            let (new, similar) = components.split(&values_components_names)?;
            if new.count() * similar.count() != components.count() {
                return Err(Error::InvalidParameter(
                    "gradient component labels must contain all possible combinations \
                    of gradient specific components and the value components".into()
                ));
            }

            if similar != **self.values.components() {
                return Err(Error::InvalidParameter(
                    "gradient component labels must contain the same entries \
                    as the values component labels for the shared variables".into()
                ));
            }
        }

        let features = Arc::clone(self.values.features());
        check_data_label_shape(
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

        let mut features = LabelsBuilder::new(vec!["f"]);
        features.add(vec![LabelValue::new(3)]);
        features.add(vec![LabelValue::new(6)]);

        let mut block = Block::new(
            TestArray::new((4, 3, 2)),
            samples.finish(),
            Arc::new(components.finish()),
            Arc::new(features.finish()),
        ).unwrap();
        assert!(block.gradients().is_empty());

        let mut gradient_samples = LabelsBuilder::new(vec!["sample", "bar"]);
        gradient_samples.add(vec![LabelValue::new(0), LabelValue::new(0)]);
        gradient_samples.add(vec![LabelValue::new(1), LabelValue::new(1)]);
        gradient_samples.add(vec![LabelValue::new(2), LabelValue::new(-2)]);
        gradient_samples.add(vec![LabelValue::new(3), LabelValue::new(-2)]);
        gradient_samples.add(vec![LabelValue::new(3), LabelValue::new(1)]);

        block.add_gradient(
            "first",
            TestArray::new((5, 3, 2)),
            gradient_samples.finish(),
            None
        ).unwrap();

        // gradient with different components
        let mut components = LabelsBuilder::new(vec!["additional", "c", "d"]);
        components.add(vec![LabelValue::new(0), LabelValue::new(-1), LabelValue::new(-4)]);
        components.add(vec![LabelValue::new(0), LabelValue::new(-2), LabelValue::new(-5)]);
        components.add(vec![LabelValue::new(0), LabelValue::new(-3), LabelValue::new(-6)]);
        components.add(vec![LabelValue::new(1), LabelValue::new(-1), LabelValue::new(-4)]);
        components.add(vec![LabelValue::new(1), LabelValue::new(-2), LabelValue::new(-5)]);
        components.add(vec![LabelValue::new(1), LabelValue::new(-3), LabelValue::new(-6)]);
        components.add(vec![LabelValue::new(2), LabelValue::new(-1), LabelValue::new(-4)]);
        components.add(vec![LabelValue::new(2), LabelValue::new(-2), LabelValue::new(-5)]);
        components.add(vec![LabelValue::new(2), LabelValue::new(-3), LabelValue::new(-6)]);

        let mut gradient_samples = LabelsBuilder::new(vec!["sample", "bar"]);
        gradient_samples.add(vec![LabelValue::new(0), LabelValue::new(0)]);
        gradient_samples.add(vec![LabelValue::new(1), LabelValue::new(1)]);
        gradient_samples.add(vec![LabelValue::new(2), LabelValue::new(-2)]);
        gradient_samples.add(vec![LabelValue::new(3), LabelValue::new(-2)]);
        gradient_samples.add(vec![LabelValue::new(3), LabelValue::new(1)]);

        block.add_gradient(
            "second",
            TestArray::new((5, 9, 2)),
            gradient_samples.finish(),
            Arc::new(components.finish()),
        ).unwrap();

        let mut gradients_list = block.gradients().keys().collect::<Vec<_>>();
        gradients_list.sort_unstable();
        assert_eq!(gradients_list, ["first", "second"]);
        assert!(block.gradients.contains_key("first"));
        assert!(block.gradients.contains_key("second"));

        assert!(block.gradients().get("bar").is_none());

        let basic_block = block.gradients().get("first").unwrap();
        assert_eq!(basic_block.samples().names(), ["sample", "bar"]);
        assert_eq!(basic_block.components().names(), ["c", "d"]);
        assert_eq!(basic_block.features().names(), ["f"]);

        let basic_block = block.gradients().get("second").unwrap();
        assert_eq!(basic_block.samples().names(), ["sample", "bar"]);
        assert_eq!(basic_block.components().names(), ["additional", "c", "d"]);
        assert_eq!(basic_block.features().names(), ["f"]);
    }

    #[test]
    fn gradient_components_invariant_values() {
        let mut samples = LabelsBuilder::new(vec!["samples"]);
        samples.add(vec![LabelValue::new(0)]);

        let mut features = LabelsBuilder::new(vec!["features"]);
        features.add(vec![LabelValue::new(0)]);

        // special using Label::single for component, there is no need to
        // duplicate the variable name & values in gradient components
        let mut block = Block::new(
            TestArray::new((1, 1, 1)),
            samples.finish(),
            Arc::new(Labels::single()),
            Arc::new(features.finish()),
        ).unwrap();

        let mut gradient_samples = LabelsBuilder::new(vec!["sample", "parameter"]);
        gradient_samples.add(vec![LabelValue::new(0), LabelValue::new(-3)]);

        let mut components = LabelsBuilder::new(vec!["cartesian"]);
        components.add(vec![LabelValue::new(0)]);
        components.add(vec![LabelValue::new(1)]);
        components.add(vec![LabelValue::new(2)]);

        block.add_gradient(
            "parameter",
            TestArray::new((1, 3, 1)),
            gradient_samples.finish(),
            Arc::new(components.finish()),
        ).unwrap();
    }

    #[test]
    fn invalid_gradient_components() {
        let mut samples = LabelsBuilder::new(vec!["samples"]);
        samples.add(vec![LabelValue::new(0)]);

        let mut components = LabelsBuilder::new(vec!["component_1", "component_2"]);
        components.add(vec![LabelValue::new(1), LabelValue::new(-5)]);
        components.add(vec![LabelValue::new(3), LabelValue::new(4)]);

        let mut features = LabelsBuilder::new(vec!["features"]);
        features.add(vec![LabelValue::new(0)]);

        let mut block = Block::new(
            TestArray::new((1, 2, 1)),
            samples.finish(),
            Arc::new(components.finish()),
            Arc::new(features.finish()),
        ).unwrap();

        let mut gradient_samples = LabelsBuilder::new(vec!["sample", "parameter"]);
        gradient_samples.add(vec![LabelValue::new(0), LabelValue::new(-3)]);
        let gradient_samples = gradient_samples.finish();

        let mut components = LabelsBuilder::new(vec!["not_enough"]);
        components.add(vec![LabelValue::new(1)]);

        let error = block.add_gradient(
            "parameter",
            TestArray::new((1, 1, 1)),
            gradient_samples.clone(),
            Arc::new(components.finish()),
        ).unwrap_err();
        assert_eq!(format!("{}", error),
            "invalid parameter: gradient component labels must contain at \
            least the same number of variables as the value components, \
            got 1 but expected at least 2"
        );

        /**********************************************************************/

        let mut components = LabelsBuilder::new(vec!["not_enough", "component_2"]);
        components.add(vec![LabelValue::new(1), LabelValue::new(3)]);

        let error = block.add_gradient(
            "parameter",
            TestArray::new((1, 1, 1)),
            gradient_samples.clone(),
            Arc::new(components.finish()),
        ).unwrap_err();
        assert_eq!(format!("{}", error),
            "invalid parameter: gradient component labels must end with the \
            same variables as values component labels, got [not_enough, \
            component_2] but the values contains [component_1, component_2]"
        );

        /**********************************************************************/

        let mut components = LabelsBuilder::new(vec!["component_1", "component_2"]);
        components.add(vec![LabelValue::new(1), LabelValue::new(3)]);

        let error = block.add_gradient(
            "parameter",
            TestArray::new((1, 1, 1)),
            gradient_samples.clone(),
            Arc::new(components.finish()),
        ).unwrap_err();
        assert_eq!(format!("{}", error),
            "invalid parameter: gradient component labels must contain the same \
            entries as the values component labels for the shared variables"
        );

        /**********************************************************************/

        let mut components = LabelsBuilder::new(vec!["new", "component_1", "component_2"]);
        components.add(vec![LabelValue::new(0), LabelValue::new(1), LabelValue::new(-5)]);
        components.add(vec![LabelValue::new(0), LabelValue::new(3), LabelValue::new(4)]);
        components.add(vec![LabelValue::new(1), LabelValue::new(1), LabelValue::new(-5)]);

        let error = block.add_gradient(
            "parameter",
            TestArray::new((1, 1, 1)),
            gradient_samples,
            Arc::new(components.finish()),
        ).unwrap_err();
        assert_eq!(format!("{}", error),
            "invalid parameter: gradient component labels must contain all \
            possible combinations of gradient specific components and the \
            value components"
        );
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
                aml_array_t::new(Box::new(Array3::from_elem((3, 1, 3), 11.0))),
                grad_samples.clone(),
                None,
            ).unwrap();

            let mut components = LabelsBuilder::new(vec!["another", "components"]);
            components.add(vec![LabelValue::new(0), LabelValue::new(0)]);
            components.add(vec![LabelValue::new(1), LabelValue::new(0)]);
            components.add(vec![LabelValue::new(2), LabelValue::new(0)]);

            block.add_gradient(
                "with_components",
                aml_array_t::new(Box::new(Array3::from_elem((3, 3, 3), 111.0))),
                grad_samples,
                Arc::new(components.finish()),
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

            let gradient = block.gradients().get("parameter").unwrap();
            assert_eq!(gradient.samples().names(), ["sample", "parameter"]);
            assert_eq!(gradient.samples().count(), 3);
            assert_eq!(gradient.samples()[0], [LabelValue::new(0), LabelValue::new(2)]);
            assert_eq!(gradient.samples()[1], [LabelValue::new(0), LabelValue::new(3)]);
            assert_eq!(gradient.samples()[2], [LabelValue::new(1), LabelValue::new(2)]);

            assert_eq!(gradient.data.as_array(), Array3::from_elem((3, 1, 3), 11.0));

            let gradient = block.gradients().get("with_components").unwrap();
            assert_eq!(gradient.data.as_array(), Array3::from_elem((3, 3, 3), 111.0));
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
                aml_array_t::new(Box::new(Array3::from_elem((3, 9, 1), 11.0))),
                grad_samples.clone(),
                None,
            ).unwrap();

            let mut components = LabelsBuilder::new(vec!["another", "component_1", "component_2"]);
            components.add(vec![LabelValue::new(0), LabelValue::new(-1), LabelValue::new(0)]);
            components.add(vec![LabelValue::new(0), LabelValue::new(-1), LabelValue::new(1)]);
            components.add(vec![LabelValue::new(0), LabelValue::new(-1), LabelValue::new(2)]);
            components.add(vec![LabelValue::new(0), LabelValue::new(0), LabelValue::new(0)]);
            components.add(vec![LabelValue::new(0), LabelValue::new(0), LabelValue::new(1)]);
            components.add(vec![LabelValue::new(0), LabelValue::new(0), LabelValue::new(2)]);
            components.add(vec![LabelValue::new(0), LabelValue::new(1), LabelValue::new(0)]);
            components.add(vec![LabelValue::new(0), LabelValue::new(1), LabelValue::new(1)]);
            components.add(vec![LabelValue::new(0), LabelValue::new(1), LabelValue::new(2)]);

            components.add(vec![LabelValue::new(1), LabelValue::new(-1), LabelValue::new(0)]);
            components.add(vec![LabelValue::new(1), LabelValue::new(-1), LabelValue::new(1)]);
            components.add(vec![LabelValue::new(1), LabelValue::new(-1), LabelValue::new(2)]);
            components.add(vec![LabelValue::new(1), LabelValue::new(0), LabelValue::new(0)]);
            components.add(vec![LabelValue::new(1), LabelValue::new(0), LabelValue::new(1)]);
            components.add(vec![LabelValue::new(1), LabelValue::new(0), LabelValue::new(2)]);
            components.add(vec![LabelValue::new(1), LabelValue::new(1), LabelValue::new(0)]);
            components.add(vec![LabelValue::new(1), LabelValue::new(1), LabelValue::new(1)]);
            components.add(vec![LabelValue::new(1), LabelValue::new(1), LabelValue::new(2)]);

            block.add_gradient(
                "with_components",
                aml_array_t::new(Box::new(Array3::from_elem((3, 18, 1), 111.0))),
                grad_samples,
                Arc::new(components.finish()),
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

            let gradient = block.gradients().get("parameter").unwrap();
            assert_eq!(gradient.samples().names(), ["sample", "parameter"]);
            assert_eq!(gradient.samples().count(), 3);
            assert_eq!(gradient.samples()[0], [LabelValue::new(0), LabelValue::new(2)]);
            assert_eq!(gradient.samples()[1], [LabelValue::new(0), LabelValue::new(3)]);
            assert_eq!(gradient.samples()[2], [LabelValue::new(1), LabelValue::new(2)]);

            assert_eq!(gradient.data.as_array(), Array3::from_elem((3, 3, 3), 11.0));

            let gradient = block.gradients().get("with_components").unwrap();
            assert_eq!(gradient.components().names(), ["another", "component_2"]);
            assert_eq!(gradient.components().count(), 6);
            assert_eq!(gradient.components()[0], [LabelValue::new(0), LabelValue::new(0)]);
            assert_eq!(gradient.components()[1], [LabelValue::new(0), LabelValue::new(1)]);
            assert_eq!(gradient.components()[2], [LabelValue::new(0), LabelValue::new(2)]);
            assert_eq!(gradient.components()[3], [LabelValue::new(1), LabelValue::new(0)]);
            assert_eq!(gradient.components()[4], [LabelValue::new(1), LabelValue::new(1)]);
            assert_eq!(gradient.components()[5], [LabelValue::new(1), LabelValue::new(2)]);

            assert_eq!(gradient.data.as_array(), Array3::from_elem((3, 6, 3), 111.0));
        }
    }
}
