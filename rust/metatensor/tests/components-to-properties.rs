use metatensor::{TensorBlock, TensorMap, Labels};

mod utils;
use utils::example_labels;

#[test]
fn one_component() {
    let mut block = TensorBlock::new(
        ndarray::Array::from_elem(vec![3, 2, 3], 1.0),
        &example_labels(vec!["samples"], vec![[0], [1], [2]]),
        &[example_labels(vec!["components"], vec![[0], [1]])],
        &example_labels(vec!["properties"], vec![[0], [1], [2]]),
    ).unwrap();

    let gradient = TensorBlock::new(
        ndarray::Array::from_elem(vec![2, 2, 3], 11.0),
        &example_labels(vec!["sample", "parameter"], vec![[0, 2], [1, 2]]),
        &[example_labels(vec!["components"], vec![[0], [1]])],
        &example_labels(vec!["properties"], vec![[0], [1], [2]]),
    ).unwrap();
    block.add_gradient("parameter", gradient).unwrap();

    let tensor = TensorMap::new(Labels::single(), vec![block]).unwrap();
    let tensor = tensor.components_to_properties(&["components"]).unwrap();

    let block = tensor.block_by_id(0);
    let samples = block.samples();
    assert_eq!(samples.names(), ["samples"]);
    assert_eq!(samples.count(), 3);
    assert_eq!(samples.to_cpu()[0], [0]);
    assert_eq!(samples.to_cpu()[1], [1]);
    assert_eq!(samples.to_cpu()[2], [2]);

    assert_eq!(block.components().len(), 0);

    let properties = block.properties();
    assert_eq!(properties.names(), ["components", "properties"]);
    assert_eq!(properties.count(), 6);
    assert_eq!(properties.to_cpu()[0], [0, 0]);
    assert_eq!(properties.to_cpu()[1], [0, 1]);
    assert_eq!(properties.to_cpu()[2], [0, 2]);
    assert_eq!(properties.to_cpu()[3], [1, 0]);
    assert_eq!(properties.to_cpu()[4], [1, 1]);
    assert_eq!(properties.to_cpu()[5], [1, 2]);

    let values = block.values().to_ndarray_lock::<f64>().read().unwrap();
    assert_eq!(*values, ndarray::Array::from_elem(vec![3, 6], 1.0));

    let gradient = block.gradient("parameter").unwrap();
    let samples = gradient.samples();
    assert_eq!(samples.names(), ["sample", "parameter"]);
    assert_eq!(samples.count(), 2);
    assert_eq!(samples.to_cpu()[0], [0, 2]);
    assert_eq!(samples.to_cpu()[1], [1, 2]);

    let values = gradient.values().to_ndarray_lock::<f64>().read().unwrap();
    assert_eq!(*values, ndarray::Array::from_elem(vec![2, 6], 11.0));
}

#[test]
fn multiple_components() {
    let data = ndarray::Array::from_shape_vec(vec![2, 2, 3, 2], vec![
        1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0,
        -1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.0, 4.0, -5.0, 5.0, -6.0, 6.0,
    ]).unwrap();

    let components = [
        example_labels(vec!["component_1"], vec![[0], [1]]),
        example_labels(vec!["component_2"], vec![[0], [1], [2]]),
    ];
    let properties = example_labels(vec!["properties"], vec![[0], [1]]);

    let mut block = TensorBlock::new(
        data,
        &example_labels(vec!["samples"], vec![[0], [1]]),
        &components,
        &properties,
    ).unwrap();

    let gradient = TensorBlock::new(
        ndarray::Array::from_elem(vec![3, 2, 3, 2], 11.0),
        &example_labels(vec!["sample", "parameter"], vec![[0, 2], [0, 3], [1, 2]]),
        &components,
        &properties,
    ).unwrap();

    block.add_gradient("parameter", gradient).unwrap();

    let tensor = TensorMap::new(Labels::single(), vec![block]).unwrap();
    let tensor = tensor.components_to_properties(&["component_1"]).unwrap();

    let block = tensor.block_by_id(0);
    let samples = block.samples();
    assert_eq!(samples.names(), ["samples"]);
    assert_eq!(samples.count(), 2);
    assert_eq!(samples.to_cpu()[0], [0]);
    assert_eq!(samples.to_cpu()[1], [1]);

    assert_eq!(block.components().len(), 1);
    let component = &block.components()[0];
    assert_eq!(component.names(), ["component_2"]);
    assert_eq!(component.count(), 3);
    assert_eq!(component.to_cpu()[0], [0]);
    assert_eq!(component.to_cpu()[1], [1]);
    assert_eq!(component.to_cpu()[2], [2]);

    let properties = block.properties();
    assert_eq!(properties.names(), ["component_1", "properties"]);
    assert_eq!(properties.count(), 4);
    assert_eq!(properties.to_cpu()[0], [0, 0]);
    assert_eq!(properties.to_cpu()[1], [0, 1]);
    assert_eq!(properties.to_cpu()[2], [1, 0]);
    assert_eq!(properties.to_cpu()[3], [1, 1]);

    let expected = ndarray::Array::from_shape_vec(vec![2, 3, 4], vec![
        1.0, 1.0, 4.0, 4.0, 2.0, 2.0, 5.0, 5.0, 3.0, 3.0, 6.0, 6.0,
        -1.0, 1.0, -4.0, 4.0, -2.0, 2.0, -5.0, 5.0, -3.0, 3.0, -6.0, 6.0,
    ]).unwrap();
    let values = block.values().to_ndarray_lock::<f64>().read().unwrap();
    assert_eq!(*values, expected);

    let gradient = block.gradient("parameter").unwrap();
    let samples = gradient.samples();
    assert_eq!(samples.names(), ["sample", "parameter"]);
    assert_eq!(samples.count(), 3);
    assert_eq!(samples.to_cpu()[0], [0, 2]);
    assert_eq!(samples.to_cpu()[1], [0, 3]);
    assert_eq!(samples.to_cpu()[2], [1, 2]);

    let values = gradient.values().to_ndarray_lock::<f64>().read().unwrap();
    assert_eq!(*values, ndarray::Array::from_elem(vec![3, 3, 4], 11.0));
}
