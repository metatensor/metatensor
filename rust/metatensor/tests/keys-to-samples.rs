use metatensor::{Labels, TensorMap};

mod utils;
use utils::{example_tensor, example_block, make_fill_value};

#[test]
fn sorted_samples() {
    let keys_to_move = Labels::empty(vec!["key_2"]);
    let tensor = example_tensor().keys_to_samples(&keys_to_move, make_fill_value(0.0), true).unwrap();

    let keys = tensor.keys();
    assert_eq!(keys.count(), 3);
    assert_eq!(keys.names(), ["key_1"]);
    assert_eq!(keys.to_cpu()[0], [0]);
    assert_eq!(keys.to_cpu()[1], [1]);
    assert_eq!(keys.to_cpu()[2], [2]);

    // The first two blocks are not modified
    let block_1 = tensor.block_by_id(0);
    let values = block_1.values().to_ndarray_lock::<f64>().read().unwrap();
    assert_eq!(*values, ndarray::Array::from_elem(vec![3, 1, 1], 1.0));

    let samples = block_1.samples();
    assert_eq!(samples.count(), 3);
    assert_eq!(samples.to_cpu()[0], [0, 0]);
    assert_eq!(samples.to_cpu()[1], [2, 0]);
    assert_eq!(samples.to_cpu()[2], [4, 0]);

    let block_2 = tensor.block_by_id(1);
    let values = block_2.values().to_ndarray_lock::<f64>().read().unwrap();
    assert_eq!(*values, ndarray::Array::from_elem(vec![3, 1, 3], 2.0));

    let samples = block_2.samples();
    assert_eq!(samples.count(), 3);
    assert_eq!(samples.to_cpu()[0], [0, 0]);
    assert_eq!(samples.to_cpu()[1], [1, 0]);
    assert_eq!(samples.to_cpu()[2], [3, 0]);

    // The new third block contains the old third and fourth blocks merged
    let block_3 = tensor.block_by_id(2);
    let samples = block_3.samples();
    assert_eq!(samples.names(), ["samples", "key_2"]);
    assert_eq!(samples.count(), 8);
    assert_eq!(samples.to_cpu()[0], [0, 2]);
    assert_eq!(samples.to_cpu()[1], [0, 3]);
    assert_eq!(samples.to_cpu()[2], [1, 3]);
    assert_eq!(samples.to_cpu()[3], [2, 3]);
    assert_eq!(samples.to_cpu()[4], [3, 2]);
    assert_eq!(samples.to_cpu()[5], [5, 3]);
    assert_eq!(samples.to_cpu()[6], [6, 2]);
    assert_eq!(samples.to_cpu()[7], [8, 2]);

    assert_eq!(block_3.components().len(), 1);
    let component = &block_3.components()[0];

    assert_eq!(component.names(), ["components"]);
    assert_eq!(component.count(), 3);
    assert_eq!(component.to_cpu()[0], [0]);
    assert_eq!(component.to_cpu()[1], [1]);
    assert_eq!(component.to_cpu()[2], [2]);

    let properties = block_3.properties();
    assert_eq!(properties.names(), ["properties"]);
    assert_eq!(properties.count(), 1);
    assert_eq!(properties.to_cpu()[0], [0]);

    let expected = ndarray::Array::from_shape_vec(vec![8, 3, 1], vec![
        3.0, 3.0, 3.0,
        4.0, 4.0, 4.0,
        4.0, 4.0, 4.0,
        4.0, 4.0, 4.0,
        3.0, 3.0, 3.0,
        4.0, 4.0, 4.0,
        3.0, 3.0, 3.0,
        3.0, 3.0, 3.0,
    ]).unwrap();
    let values = block_3.values().to_ndarray_lock::<f64>().read().unwrap();
    assert_eq!(*values, expected);

    let gradient_3 = block_3.gradient("parameter").unwrap();
    let samples = gradient_3.samples();
    assert_eq!(samples.names(), ["sample", "parameter"]);
    assert_eq!(samples.count(), 3);
    assert_eq!(samples.to_cpu()[0], [1, 1]);
    assert_eq!(samples.to_cpu()[1], [4, -2]);
    assert_eq!(samples.to_cpu()[2], [5, 3]);

    let expected = ndarray::Array::from_shape_vec(vec![3, 3, 1], vec![
        14.0, 14.0, 14.0,
        13.0, 13.0, 13.0,
        14.0, 14.0, 14.0,
    ]).unwrap();
    let values = gradient_3.values().to_ndarray_lock::<f64>().read().unwrap();
    assert_eq!(*values, expected);
}

#[test]
fn unsorted_samples() {
    let keys_to_move = Labels::empty(vec!["key_2"]);
    let tensor = example_tensor().keys_to_samples(&keys_to_move, make_fill_value(0.0), false).unwrap();

    let block_3 = tensor.block_by_id(2);
    let samples = block_3.samples();
    assert_eq!(samples.names(), ["samples", "key_2"]);
    assert_eq!(samples.count(), 8);
    assert_eq!(samples.to_cpu()[0], [0, 2]);
    assert_eq!(samples.to_cpu()[1], [3, 2]);
    assert_eq!(samples.to_cpu()[2], [6, 2]);
    assert_eq!(samples.to_cpu()[3], [8, 2]);
    assert_eq!(samples.to_cpu()[4], [0, 3]);
    assert_eq!(samples.to_cpu()[5], [1, 3]);
    assert_eq!(samples.to_cpu()[6], [2, 3]);
    assert_eq!(samples.to_cpu()[7], [5, 3]);
}

#[test]
fn user_provided_entries() {
    let keys_to_move = Labels::new(["key_2"], &[[3]]);
    let result = example_tensor().keys_to_samples(&keys_to_move, make_fill_value(0.0), false);

    assert_eq!(
        result.unwrap_err().message,
        "invalid parameter: user provided values for the keys to move is \
        not yet implemented, `keys_to_move` should not contain any entry \
        when calling keys_to_samples"
    );
}

#[test]
#[allow(clippy::vec_init_then_push)]
fn empty_samples() {
    let mut blocks = Vec::new();
    blocks.push(example_block(
        /* samples          */ vec![[0], [2], [4]],
        /* components       */ vec![[0]],
        /* properties       */ vec![[0], [1], [2], [3]],
        /* gradient_samples */ vec![],
        /* values           */ 1.0,
        /* gradient_values  */ 11.0,
    ));

    blocks.push(example_block(
        /* samples          */ vec![],
        /* components       */ vec![[0]],
        /* properties       */ vec![[0], [1], [2], [3]],
        /* gradient_samples */ vec![],
        /* values           */ 0.0,
        /* gradient_values  */ 0.0,
    ));
    let keys = Labels::new(
        ["key_1", "key_2"],
        &[[0, 0], [1, 0]]
    );

    let tensor = TensorMap::new(keys, blocks).unwrap();

    let keys_to_move = Labels::empty(vec!["key_1"]);
    let tensor = tensor.keys_to_samples(&keys_to_move, make_fill_value(0.0), true).unwrap();

    assert_eq!(
        tensor.block_by_id(0).samples(),
        Labels::new(["samples", "key_1"], &[
            [0, 0],
            [2, 0],
            [4, 0],
        ])
    );
}
