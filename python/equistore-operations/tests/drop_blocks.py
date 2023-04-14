import os
import re

import numpy as np
import pytest

import equistore
from equistore import Labels, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


# ===== Fixtures and helper functions =====


@pytest.fixture
def test_tensor_map() -> TensorMap:
    return equistore.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        # the npz is using DEFLATE compression, equistore only supports STORED
        use_numpy=True,
    )


def check_consistency(tensor1: TensorMap, tensor2: TensorMap):
    # check if two TensorMaps contain the same information
    if tensor1.keys.names != tensor2.keys.names:
        raise ValueError("the two tensors have different key names")

    keys = tensor2.keys
    for key in keys:
        if tensor2[key] != tensor1[key]:
            raise ValueError(f"the blocks indexed by key {key} are not equivalent")


# ===== Tests =====


def test_drop_all(test_tensor_map):
    # test the behavior when all the keys are dropped
    # expected behavior: empty TensorMap
    keys = test_tensor_map.keys
    tensor = equistore.drop_blocks(test_tensor_map, keys)
    empty_tensor = TensorMap(keys=Labels.empty(keys.names), blocks=[])
    assert equistore.equal(tensor, empty_tensor)


def test_drop_first(test_tensor_map):
    # test the behavior when the first key is dropped
    ref_keys = test_tensor_map.keys[1:]
    ref_blocks = [test_tensor_map[key].copy() for key in ref_keys]
    ref_tensor = TensorMap(ref_keys, ref_blocks)

    key_to_drop = test_tensor_map.keys[:1]
    new_tensor = equistore.drop_blocks(test_tensor_map, key_to_drop)
    assert equistore.equal(new_tensor, ref_tensor)
    check_consistency(test_tensor_map, new_tensor)


def test_drop_last(test_tensor_map):
    # test the behavior when the last key is dropped
    ref_keys = test_tensor_map.keys[:-1]
    ref_blocks = [test_tensor_map[key].copy() for key in ref_keys]
    ref_tensor = TensorMap(ref_keys, ref_blocks)

    key_to_drop = test_tensor_map.keys[-1:]
    new_tensor = equistore.drop_blocks(test_tensor_map, key_to_drop)
    assert equistore.equal(new_tensor, ref_tensor)
    check_consistency(test_tensor_map, new_tensor)


def test_drop_random(test_tensor_map):
    # test the behavior when an existing  random key is dropped
    number_blocks = len(test_tensor_map.blocks())
    index_to_remove = np.random.randint(1, number_blocks - 1)

    names = test_tensor_map.keys.names
    values = []
    for i in range(index_to_remove):
        values += [tuple(test_tensor_map.keys[i])]
    for i in range(index_to_remove + 1, number_blocks):
        values += [tuple(test_tensor_map.keys[i])]
    values = np.array(values)
    ref_keys = Labels(names, values)
    ref_blocks = [test_tensor_map[key].copy() for key in ref_keys]
    ref_tensor = TensorMap(ref_keys, ref_blocks)

    key_to_drop = Labels(
        test_tensor_map.keys[index_to_remove].dtype.names,
        np.array([tuple(test_tensor_map.keys[index_to_remove])]),
    )
    new_tensor = equistore.drop_blocks(test_tensor_map, key_to_drop)
    assert equistore.equal(new_tensor, ref_tensor)
    check_consistency(test_tensor_map, new_tensor)


def test_drop_two_random_keys(test_tensor_map):
    # test the behavior when two random keys are dropped
    number_blocks = len(test_tensor_map.blocks())
    indices_to_drop = np.random.randint(0, number_blocks, 2)
    keys_to_drop = test_tensor_map.keys[indices_to_drop]

    ref_keys = np.setdiff1d(test_tensor_map.keys, keys_to_drop)
    ref_blocks = [test_tensor_map[key].copy() for key in ref_keys]
    ref_tensor = TensorMap(ref_keys, ref_blocks)

    new_tensor = equistore.drop_blocks(test_tensor_map, keys_to_drop)
    assert equistore.equal(new_tensor, ref_tensor)
    check_consistency(test_tensor_map, new_tensor)


def test_drop_nothing(test_tensor_map):
    # test when an emty key is dropped
    empty_key = Labels.empty(test_tensor_map.keys.names)
    new_tensor = equistore.drop_blocks(test_tensor_map, empty_key)
    assert new_tensor == test_tensor_map


def test_not_existent(test_tensor_map):
    # test when keys that don't appear in the TensorMap are dropped
    non_existent_key = Labels(
        test_tensor_map.keys.names, np.array([[-1, -1, -1], [0, 0, 0]])
    )
    error_msg = (
        "some keys in `keys` are not present in `tensor`."
        f" Non-existent keys: {non_existent_key}"
    )
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        equistore.drop_blocks(test_tensor_map, non_existent_key)


def test_copy_flag(test_tensor_map):
    """
    Checks that settings the copy flag to true or false gives the same result.
    Also checks that inplace modifications of the underlying data doesn't affect
    the copied returned tensor, but it does for the uncopied version.
    """
    # Define some keys to drop
    keys_to_drop = test_tensor_map.keys[:5]

    # Drop blocks with copy=True flag
    new_tensor_copied = equistore.drop_blocks(test_tensor_map, keys_to_drop, copy=True)

    # Drop blocks with copy=False flag
    new_tensor_not_copied = equistore.drop_blocks(
        test_tensor_map, keys_to_drop, copy=False
    )

    # Check that the resulting tensormaps are equal whether or not they are
    # copied
    assert equistore.equal(new_tensor_copied, new_tensor_not_copied)
    # check_consistency(new_tensor_copied, new_tensor_not_copied)

    # Now modify the original tensor's block values in place
    for block in test_tensor_map.blocks():
        block.values[:] += 3.14

    # The copied tensor's values should not have been edited
    for key in new_tensor_copied.keys:
        assert np.all(new_tensor_copied[key].values != test_tensor_map[key].values)

    # But the uncopied tensor's values should have been
    for key in new_tensor_not_copied.keys:
        assert np.all(new_tensor_not_copied[key].values == test_tensor_map[key].values)
        assert np.all(
            new_tensor_not_copied[key].values == new_tensor_copied[key].values[:] + 3.14
        )
