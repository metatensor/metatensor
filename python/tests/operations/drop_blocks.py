import os

import numpy as np

import pytest
import equistore
from equistore import TensorMap, Labels


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


class TestDropBlock:
    # this test will fail for now because of issue #175
    def drop_all(self):
        tensor = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        keys = tensor.keys
        tensor = equistore.drop_blocks(tensor, keys)
        empty_tensor = TensorMap(keys=Labels.empty(keys.names), blocks=[])
        assert equistore.equal(tensor, empty_tensor)

    def test_drop_first(self):
        tensor = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        ref_keys = tensor.keys[1:]
        ref_blocks = [tensor[key].copy() for key in ref_keys]
        ref_tensor = TensorMap(ref_keys, ref_blocks)

        key_to_drop = tensor.keys[:1]
        new_tensor = equistore.drop_blocks(tensor, key_to_drop)
        assert equistore.equal(new_tensor, ref_tensor)
        check_consistency(tensor, new_tensor)

    def test_drop_last(self):
        tensor = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        ref_keys = tensor.keys[:-1]
        ref_blocks = [tensor[key].copy() for key in ref_keys]
        ref_tensor = TensorMap(ref_keys, ref_blocks)

        key_to_drop = tensor.keys[-1:]
        new_tensor = equistore.drop_blocks(tensor, key_to_drop)
        assert equistore.equal(new_tensor, ref_tensor)
        check_consistency(tensor, new_tensor)

    def test_drop_random(self):
        tensor = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )

        number_blocks = len(tensor.blocks())
        index_to_remove = np.random.randint(1, number_blocks - 1)

        names = tensor.keys.names
        values = []
        for i in range(index_to_remove):
            values += [tuple(tensor.keys[i])]
        for i in range(index_to_remove + 1, number_blocks):
            values += [tuple(tensor.keys[i])]
        values = np.array(values)
        ref_keys = Labels(names, values)
        ref_blocks = [tensor[key].copy() for key in ref_keys]
        ref_tensor = TensorMap(ref_keys, ref_blocks)

        key_to_drop = Labels(
            tensor.keys[index_to_remove].dtype.names,
            np.array([tuple(tensor.keys[index_to_remove])]),
        )
        new_tensor = equistore.drop_blocks(tensor, key_to_drop)
        assert equistore.equal(new_tensor, ref_tensor)
        check_consistency(tensor, new_tensor)

    def test_drop_two_random_keys(self):
        tensor = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        number_blocks = len(tensor.blocks())
        indices_to_drop = np.random.randint(0, number_blocks, 2)
        keys_to_drop = tensor.keys[indices_to_drop]

        ref_keys = np.setdiff1d(tensor.keys, keys_to_drop)
        ref_blocks = [tensor[key].copy() for key in ref_keys]
        ref_tensor = TensorMap(ref_keys, ref_blocks)

        new_tensor = equistore.drop_blocks(tensor, keys_to_drop)
        assert equistore.equal(new_tensor, ref_tensor)
        check_consistency(tensor, new_tensor)

    def test_drop_nothing(self):
        tensor = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        empty_key = Labels.empty(tensor.keys.names)
        new_tensor = equistore.drop_blocks(tensor, empty_key)
        assert new_tensor == tensor

    def test_not_existent(self):
        tensor = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        non_existent_key = Labels(tensor.keys.names, np.array([[0, 0, 0]]))
        with pytest.raises(ValueError):
            equistore.drop_blocks(tensor, non_existent_key)


def check_consistency(tensor1: TensorMap, tensor2: TensorMap):
    if tensor1.keys.names != tensor2.keys.names:
        raise ValueError("the two tensors have different labels")

    keys = tensor2.keys
    for key in keys:
        if tensor2[key] != tensor1[key]:
            raise ValueError("tensor2 has values that are not present in tensor1")
