use std::sync::Arc;
use std::ffi::CString;
use std::collections::{HashMap, BTreeSet};

use crate::labels::LabelValue;
use crate::utils::ConstCString;
use crate::Labels;
use crate::{mts_array_t, get_data_origin};
use crate::Error;

/// A `Vec` which can not be modified
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImmutableVec<T>(Vec<T>);

impl<T> std::ops::Deref for ImmutableVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T> IntoIterator for &'a ImmutableVec<T> {
    type Item = &'a T;

    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

fn check_data_and_labels(
    context: &str,
    values: &mts_array_t,
    samples: &Labels,
    components: &[Arc<Labels>],
    properties: &Labels,
) -> Result<(), Error> {
    let shape = values.shape()?;

    if shape.len() != components.len() + 2 {
        return Err(Error::InvalidParameter(format!(
            "{}: the array has {} dimensions, but we have {} separate labels \
            (1 for samples, {} for components and 1 for properties)",
            context, shape.len(), components.len() + 2, components.len()
        )));
    }

    if shape[0] != samples.count() {
        return Err(Error::InvalidParameter(format!(
            "{}: the array shape along axis 0 is {} but we have {} sample label entries",
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

    let mut axis = 1;
    for component in components {
        if shape[axis] != component.count() {
            return Err(Error::InvalidParameter(format!(
                "{}: the array shape along axis {} is {} but we have {} entries \
                for the corresponding component",
                context, axis, shape[axis], component.count(),
            )));
        }

        axis += 1;
    }

    if shape[axis] != properties.count() {
        return Err(Error::InvalidParameter(format!(
            "{}: the array shape along axis {} is {} but we have {} property label entries",
            context, axis, shape[axis], properties.count()
        )));
    }

    Ok(())
}

fn check_component_labels(components: &[Arc<Labels>]) -> Result<(), Error> {
    for (i, component) in components.iter().enumerate() {
        if component.size() != 1 {
            return Err(Error::InvalidParameter(format!(
                "component labels must have a single dimension, got {}: [{}] for component {}",
                component.size(), component.names().join(", "), i
            )));
        }

        if component.is_empty() {
            return Err(Error::InvalidParameter(format!(
                "component '{}' must contain at least one entry, got 0",
                component.names()[0]
            )));
        }
    }
    Ok(())
}

/// A single block in a `TensorMap`, containing both data & the corresponding
/// metadata.
///
/// Optionally gradients of the data w.r.t. any relevant quantity can be stored
/// as additional `TensorBlock` inside this one.
#[derive(Debug)]
pub struct TensorBlock {
    pub values: mts_array_t,
    pub samples: Arc<Labels>,
    pub components: ImmutableVec<Arc<Labels>>,
    pub properties: Arc<Labels>,
    gradients: HashMap<String, TensorBlock>,
    // all the keys from `self.gradients`, as C-compatible strings
    gradient_parameters: Vec<ConstCString>,
}

impl TensorBlock {
    /// Create a new `TensorBlock` containing the given values, described by the
    /// `samples`, `components`, and `properties` labels. The block is
    /// initialized without any gradients.
    pub fn new(
        values: mts_array_t,
        samples: Arc<Labels>,
        components: Vec<Arc<Labels>>,
        properties: Arc<Labels>,
    ) -> Result<TensorBlock, Error> {
        check_data_and_labels(
            "data and labels don't match", &values, &samples, &components, &properties
        )?;
        check_component_labels(&components)?;
        let components = ImmutableVec(components);
        Ok(TensorBlock {
            values,
            samples,
            components,
            properties,
            gradients: HashMap::new(),
            gradient_parameters: Vec::new(),
        })
    }

    /// Try to copy this `TensorBlock`. This can fail if we are unable to copy
    /// one of the underlying `mts_array_t` data arrays
    pub fn try_clone(&self) -> Result<TensorBlock, Error> {
        // Try to clone the values
        let values = self.values.try_clone()?;

        // Try to clone all gradient blocks
        let mut gradients = HashMap::new();
        for (gradient_parameter, gradient_block) in &self.gradients {
            gradients.insert(gradient_parameter.clone(), gradient_block.try_clone()?);
        }
        let gradient_parameters = self.gradient_parameters.clone();

        Ok(TensorBlock {
            values,
            samples: Arc::clone(&self.samples),
            components: self.components.clone(),
            properties: Arc::clone(&self.properties),
            gradients,
            gradient_parameters
        })
    }

    /// Get all gradients defined in this block
    pub fn gradients(&self) -> &HashMap<String, TensorBlock> {
        &self.gradients
    }

    /// Get the data and metadata for the gradient with respect to the given
    /// parameter in this block, if it exists.
    pub fn gradient(&self, parameter: &str) -> Option<&TensorBlock> {
        self.gradients.get(parameter)
    }

    /// Get the data and metadata for the gradient with respect to the given
    /// parameter in this block, if it exists.
    pub fn gradient_mut(&mut self, parameter: &str) -> Option<&mut TensorBlock> {
        self.gradients.get_mut(parameter)
    }

    /// Get the list of gradients in this block for the C API
    pub fn gradient_parameters_c(&self) -> &[ConstCString] {
        &self.gradient_parameters
    }

    /// Add a gradient with respect to `parameter` to this block.
    ///
    /// The gradient `data` is given as an array, and the samples and components
    /// labels must be provided. The property labels are assumed to match the
    /// ones of the values in the current block.
    ///
    /// The components labels must contain at least the same entries as the
    /// value components labels, and can prepend other components labels.
    pub fn add_gradient(
        &mut self,
        parameter: &str,
        gradient: TensorBlock
    ) -> Result<(), Error> {
        if self.gradients.contains_key(parameter) {
            return Err(Error::InvalidParameter(format!(
                "gradient with respect to '{}' already exists for this block", parameter
            )));
        }

        if gradient.values.origin()? != self.values.origin()? {
            return Err(Error::InvalidParameter(format!(
                "the gradient data has a different origin ('{}') than the value data ('{}')",
                get_data_origin(gradient.values.origin()?),
                get_data_origin(self.values.origin()?),
            )));
        }

        if gradient.samples.size() == 0 {
            return Err(Error::InvalidParameter(
                "gradients samples must have at least one dimension, named 'sample', we got none".into()
            ));
        }

        if gradient.samples.size() < 1 || gradient.samples.names()[0] != "sample" {
            return Err(Error::InvalidParameter(format!(
                "'{}' is not valid for the first dimension in the gradients \
                samples labels, it should be 'sample'", gradient.samples.names()[0]
            )));
        }

        if gradient.samples.count() > 0 {
            let mut min_sample_value = LabelValue::new(0);
            let mut max_sample_value = LabelValue::new(0);
            if gradient.samples.is_sorted() {
                // only check the first and last entry
                min_sample_value = gradient.samples[0][0];
                let last = gradient.samples.count() - 1;
                max_sample_value = gradient.samples[last][0];
            } else {
                // check everything
                for sample in &*gradient.samples {
                    let sample_value = sample[0];
                    if sample_value < min_sample_value {
                        min_sample_value = sample_value;
                    } else if sample_value > max_sample_value {
                        max_sample_value = sample_value;
                    }
                }
            }

            if min_sample_value.i32() < 0 {
                return Err(Error::InvalidParameter(format!(
                    "invalid value for the 'sample' dimension in gradient samples: \
                    all values should be positive, but we got {}", min_sample_value
                )));
            }

            if max_sample_value.usize() >= self.samples.count() {
                return Err(Error::InvalidParameter(format!(
                    "invalid value for the 'sample' dimension in gradient samples: we got \
                    {}, but the values contain {} samples", max_sample_value, self.samples.count()
                )));
            }
        }

        check_component_labels(&gradient.components)?;
        if self.components.len() > gradient.components.len() {
            return Err(Error::InvalidParameter(
                "gradients components should contain at least as many labels \
                as the values components".into()
            ));
        }

        if gradient.properties != self.properties {
            return Err(Error::InvalidParameter(
                "gradient properties must be the same as values properties".into()
            ));
        }

        let extra_gradient_components = gradient.components.len() - self.components.len();
        for (component_i, (gradient_labels, values_labels)) in gradient.components.iter()
            .skip(extra_gradient_components)
            .zip(&*self.components)
            .enumerate() {
                if gradient_labels != values_labels {
                    return Err(Error::InvalidParameter(format!(
                        "gradients and values components mismatch for values \
                        component {} (dimension name is '{}'). Components which \
                        are specific to the gradients must come first, and be \
                        followed by the exact same components as the values.",
                        component_i, values_labels.names()[0]
                    )));
                }
            }

        self.gradients.insert(parameter.into(), gradient);

        let parameter = ConstCString::new(CString::new(parameter.to_owned()).expect("invalid C string"));
        self.gradient_parameters.push(parameter);

        return Ok(());
    }

    /// Move components to properties for this block and all gradients in this
    /// block
    pub(crate) fn components_to_properties(&mut self, dimensions: &[&str]) -> Result<(), Error> {
        if dimensions.is_empty() {
            return Ok(());
        }

        let mut component_axis = None;
        for (component_i, component) in self.components.iter().enumerate() {
            if component.names() == dimensions {
                component_axis = Some(component_i);
                break;
            }
        }

        let component_axis = component_axis.ok_or_else(|| Error::InvalidParameter(format!(
            "unable to find [{}] in the components ", dimensions.join(", ")
        )))?;

        let moved_component = self.components.0.remove(component_axis);

        // construct the new property with old properties and the components
        let old_properties = &self.properties;
        let new_property_names = moved_component.names().iter()
            .chain(old_properties.names().iter())
            .copied()
            .collect::<Vec<_>>();

        let mut new_property_values = Vec::new();
        new_property_values.reserve_exact(
            moved_component.count() * moved_component.size()
            * old_properties.count() * old_properties.size()
        );
        for new_property in &*moved_component {
            for old_property in old_properties.iter() {
                new_property_values.extend_from_slice(new_property);
                new_property_values.extend_from_slice(old_property);
            }
        }
        let new_properties = unsafe {
            // SAFETY: the new property can only contain unique entries by construction
            Labels::new_unchecked_uniqueness(&new_property_names, new_property_values).expect("invalid labels")
        };

        let mut new_shape = self.values.shape()?.to_vec();
        let properties_axis = new_shape.len() - 1;
        new_shape[properties_axis] = new_properties.count();
        new_shape.remove(component_axis + 1);

        self.values.swap_axes(component_axis + 1, properties_axis - 1)?;
        self.values.reshape(&new_shape)?;

        self.properties = Arc::new(new_properties);

        // Repeat all the above for all gradient blocks
        for gradient in self.gradients.values_mut() {
            gradient.components_to_properties(dimensions)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::data::TestArray;

    use super::*;

    fn example_labels(name: &str, count: i32) -> Arc<Labels> {
        return Arc::new(Labels::new(&[name], (0..count).collect()).expect("invalid labels"));
    }

    #[test]
    fn no_components() {
        let samples = example_labels("samples", 4);
        let properties = example_labels("properties", 7);
        let values = TestArray::new(vec![4, 7]);
        let result = TensorBlock::new(values, samples.clone(), Vec::new(), properties.clone());
        assert!(result.is_ok());

        let values = TestArray::new(vec![3, 7]);
        let result = TensorBlock::new(values, samples.clone(), Vec::new(), properties.clone());
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: data and labels don't match: the array shape \
            along axis 0 is 3 but we have 4 sample label entries"
        );

        let values = TestArray::new(vec![4, 9]);
        let result = TensorBlock::new(values, samples.clone(), Vec::new(), properties.clone());
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: data and labels don't match: the array shape \
            along axis 1 is 9 but we have 7 property label entries"
        );

        let values = TestArray::new(vec![4, 1, 7]);
        let result = TensorBlock::new(values, samples, Vec::new(), properties);
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: data and labels don't match: the array has \
            3 dimensions, but we have 2 separate labels (1 for samples, 0 \
            for components and 1 for properties)"
        );
    }

    #[test]
    fn multiple_components() {
        let component_1 = example_labels("component_1", 4);
        let component_2 = example_labels("component_2", 3);

        let samples = example_labels("samples", 3);
        let properties = example_labels("properties", 2);
        let values = TestArray::new(vec![3, 4, 2]);
        let components = vec![Arc::clone(&component_1)];
        let result = TensorBlock::new(values, samples.clone(), components, properties.clone());
        assert!(result.is_ok());

        let values = TestArray::new(vec![3, 4, 3, 2]);
        let components = vec![Arc::clone(&component_1), Arc::clone(&component_2)];
        let result = TensorBlock::new(values, samples.clone(), components, properties.clone());
        assert!(result.is_ok());

        let values = TestArray::new(vec![3, 4, 2]);
        let components = vec![Arc::clone(&component_1), Arc::clone(&component_2)];
        let result = TensorBlock::new(values, samples.clone(), components, properties.clone());
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: data and labels don't match: the array has 3 \
            dimensions, but we have 4 separate labels (1 for samples, 2 for \
            components and 1 for properties)"
        );

        let values = TestArray::new(vec![3, 4, 4, 2]);
        let components = vec![Arc::clone(&component_1), Arc::clone(&component_2)];
        let result = TensorBlock::new(values, samples.clone(), components, properties.clone());
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: data and labels don't match: the array shape \
            along axis 2 is 4 but we have 3 entries for the corresponding component"
        );

        let values = TestArray::new(vec![3, 4, 4, 2]);
        let components = vec![Arc::clone(&component_1), Arc::clone(&component_1)];
        let result = TensorBlock::new(values, samples.clone(), components, properties.clone());
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: data and labels don't match: some of the \
            component names appear more than once in component labels"
        );

        let values = TestArray::new(vec![3, 1, 2]);
        let component = Arc::new(Labels::new(
            &["component_1", "component_2"],
            vec![0, 1],
        ).expect("invalid labels"));

        let result = TensorBlock::new(values, samples, vec![component], properties);
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: component labels must have a single dimension, \
            got 2: [component_1, component_2] for component 0"
        );
    }

    mod gradients {
        use super::*;

        #[test]
        fn values_without_components() {
            let samples = example_labels("samples", 4);
            let properties = example_labels("properties", 7);
            let values = TestArray::new(vec![4, 7]);
            let mut block = TensorBlock::new(values, samples, vec![], properties.clone()).unwrap();
            assert!(block.gradients().is_empty());

            let gradient_samples = Arc::new(Labels::new(
                &["sample", "foo"],
                vec![0, 0, /**/ 1, 1, /**/ 3, -2],
            ).expect("invalid labels"));

            let gradient = TensorBlock::new(
                TestArray::new(vec![3, 7]),
                gradient_samples,
                vec![],
                properties.clone(),
            ).unwrap();
            block.add_gradient("foo", gradient).unwrap();

            let gradient = TensorBlock::new(
                TestArray::new(vec![3, 5, 7]),
                example_labels("sample", 3),
                vec![example_labels("component", 5)],
                properties,
            ).unwrap();
            block.add_gradient("component", gradient).unwrap();

            let mut gradients_list = block.gradients().keys().collect::<Vec<_>>();
            gradients_list.sort_unstable();
            assert_eq!(gradients_list, ["component", "foo"]);

            let gradient_block = block.gradients().get("foo").unwrap();
            assert_eq!(gradient_block.samples.names(), ["sample", "foo"]);
            assert!(gradient_block.components.is_empty());
            assert_eq!(gradient_block.properties.names(), ["properties"]);

            let gradient_block = block.gradients().get("component").unwrap();
            assert_eq!(gradient_block.samples.names(), ["sample"]);
            assert_eq!(gradient_block.components.len(), 1);
            assert_eq!(gradient_block.components[0].names(), ["component"]);
            assert_eq!(gradient_block.properties.names(), ["properties"]);

            assert!(block.gradients().get("baz").is_none());
        }

        #[test]
        fn values_with_components() {
            let component = example_labels("component", 5);
            let properties = example_labels("properties", 7);
            let mut block = TensorBlock::new(
                TestArray::new(vec![4, 5, 7]),
                example_labels("samples", 4),
                vec![component.clone()],
                properties.clone(),
            ).unwrap();


            let gradient_samples = example_labels("sample", 3);
            let gradient = TensorBlock::new(
                TestArray::new(vec![3, 5, 7]),
                gradient_samples.clone(),
                vec![component.clone()],
                properties.clone(),
            ).unwrap();
            let result = block.add_gradient("gradient", gradient);
            assert!(result.is_ok());

            let component_2 = example_labels("component_2", 3);
            let gradient = TensorBlock::new(
                TestArray::new(vec![3, 3, 5, 7]),
                gradient_samples,
                vec![component_2, component],
                properties,
            ).unwrap();
            let result = block.add_gradient("components", gradient);
            assert!(result.is_ok());
        }
    }
}
