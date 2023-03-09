import os

import numpy as np
import pytest

import equistore
from equistore import Labels, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture
def test_tensor_map() -> TensorMap:
    return equistore.io.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        # the npz is using DEFLATE compression, equistore only supports STORED
        use_numpy=True,
    )


class TestDropBlock:
    # this test will fail for now because of issue #175
    def test_drop_all(self, test_tensor_map):
        keys = test_tensor_map.keys
        tensor = equistore.drop_blocks(test_tensor_map, keys)
        empty_tensor = TensorMap(keys=Labels.empty(keys.names), blocks=[])
        assert equistore.equal(tensor, empty_tensor)

    def test_drop_first(self, test_tensor_map):
        ref_keys = test_tensor_map.keys[1:]
        ref_blocks = [test_tensor_map[key].copy() for key in ref_keys]
        ref_tensor = TensorMap(ref_keys, ref_blocks)

        key_to_drop = test_tensor_map.keys[:1]
        new_tensor = equistore.drop_blocks(test_tensor_map, key_to_drop)
        assert equistore.equal(new_tensor, ref_tensor)
        check_consistency(test_tensor_map, new_tensor)

    def test_drop_last(self, test_tensor_map):
        ref_keys = test_tensor_map.keys[:-1]
        ref_blocks = [test_tensor_map[key].copy() for key in ref_keys]
        ref_tensor = TensorMap(ref_keys, ref_blocks)

        key_to_drop = test_tensor_map.keys[-1:]
        new_tensor = equistore.drop_blocks(test_tensor_map, key_to_drop)
        assert equistore.equal(new_tensor, ref_tensor)
        check_consistency(test_tensor_map, new_tensor)

    def test_drop_random(self, test_tensor_map):
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

    def test_drop_two_random_keys(self, test_tensor_map):
        number_blocks = len(test_tensor_map.blocks())
        indices_to_drop = np.random.randint(0, number_blocks, 2)
        keys_to_drop = test_tensor_map.keys[indices_to_drop]

        ref_keys = np.setdiff1d(test_tensor_map.keys, keys_to_drop)
        ref_blocks = [test_tensor_map[key].copy() for key in ref_keys]
        ref_tensor = TensorMap(ref_keys, ref_blocks)

        new_tensor = equistore.drop_blocks(test_tensor_map, keys_to_drop)
        assert equistore.equal(new_tensor, ref_tensor)
        check_consistency(test_tensor_map, new_tensor)

    def test_drop_nothing(self, test_tensor_map):
        empty_key = Labels.empty(test_tensor_map.keys.names)
        new_tensor = equistore.drop_blocks(test_tensor_map, empty_key)
        assert new_tensor == test_tensor_map

    def test_not_existent(self, test_tensor_map):
        non_existent_key = Labels(test_tensor_map.keys.names, np.array([[0, 0, 0]]))
        with pytest.raises(ValueError):
            equistore.drop_blocks(test_tensor_map, non_existent_key)


def check_consistency(tensor1: TensorMap, tensor2: TensorMap):
    if tensor1.keys.names != tensor2.keys.names:
        raise ValueError("the two tensors have different key names")

    keys = tensor2.keys
    for key in keys:
        if tensor2[key] != tensor1[key]:
            raise ValueError(f"the blocks indexed by key {key} are not equivalent")
