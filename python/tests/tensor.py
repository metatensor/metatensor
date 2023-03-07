import numpy as np
import pytest
from numpy.testing import assert_equal
from utils import large_tensor_map, tensor_map

import equistore


class TestTensorMap:
    @pytest.fixture
    def tensor(self):
        return tensor_map()

    @pytest.fixture
    def large_tensor(self):
        return large_tensor_map()

    def test_copy(self, tensor):
        copy = tensor.copy()
        block_1_values_id = id(tensor.block(0).values)

        del tensor

        assert id(copy.block(0).values) != block_1_values_id

        assert_equal(copy.block(0).values, np.full((3, 1, 1), 1.0))

    def test_keys(self, tensor):
        assert tensor.keys.names == ("key_1", "key_2")
        assert len(tensor.keys) == 4
        assert len(tensor) == 4
        assert tuple(tensor.keys[0]) == (0, 0)
        assert tuple(tensor.keys[1]) == (1, 0)
        assert tuple(tensor.keys[2]) == (2, 2)
        assert tuple(tensor.keys[3]) == (2, 3)

    def test_print(self, tensor):
        """
        Test routine for the print function of the TensorBlock.
        It compare the results with those in a file.
        """
        repr = tensor.__repr__()
        expected = """TensorMap with 4 blocks
keys: ['key_1' 'key_2']
          0       0
          1       0
          2       2
          2       3"""
        assert expected == repr

    def test_print_large(self, large_tensor):
        _print = large_tensor.__repr__()
        expected = """TensorMap with 12 blocks
keys: ['key_1' 'key_2']
          0       0
          1       0
          2       2
       ...
          1       5
          2       5
          3       5"""
        assert expected == _print

    def test_labels_names(self, tensor):
        assert tensor.sample_names == ("samples",)
        assert tensor.components_names == [("components",)]
        assert tensor.property_names == ("properties",)

    def test_block(self, tensor):
        # block by index
        block = tensor.block(2)
        assert_equal(block.values, np.full((4, 3, 1), 3.0))

        # block by index with __getitem__
        block = tensor[2]
        assert_equal(block.values, np.full((4, 3, 1), 3.0))

        # block by kwargs
        block = tensor.block(key_1=1, key_2=0)
        assert_equal(block.values, np.full((3, 1, 3), 2.0))

        # block by Label entry
        block = tensor.block(tensor.keys[0])
        assert_equal(block.values, np.full((3, 1, 1), 1.0))

        # block by Label entry with __getitem__
        block = tensor[tensor.keys[0]]
        assert_equal(block.values, np.full((3, 1, 1), 1.0))

        # More arguments than needed: two integers
        # by index
        with pytest.raises(
            ValueError, match="only one non-keyword argument is supported, 2 are given"
        ):
            tensor.block(3, 4)

        # 4 input with the first as integer by __getitem__
        with pytest.raises(
            ValueError, match="only one non-keyword argument is supported, 4 are given"
        ):
            tensor[3, 4, 7.0, "r"]

        # More arguments than needed: 3 Labels
        with pytest.raises(
            ValueError, match="only one non-keyword argument is supported, 3 are given"
        ):
            tensor.block(tensor.keys[0], tensor.keys[1], tensor.keys[3])

        # by __getitem__
        with pytest.raises(
            ValueError, match="only one non-keyword argument is supported, 2 are given"
        ):
            tensor[tensor.keys[1], 4]

        # 0 blocks matching criteria
        with pytest.raises(
            ValueError,
            match="Couldn't find any block matching the selection 'key_1 = 3'",
        ):
            tensor.block(key_1=3)

        # more than one block matching criteria
        with pytest.raises(
            ValueError,
            match="more than one block matched 'key_2 = 0', use `TensorMap.blocks` "
            "if you want to get all of them",
        ):
            tensor.block(key_2=0)

    def test_blocks(self, tensor):
        # block by index
        blocks = tensor.blocks(2)
        assert len(blocks) == 1
        assert_equal(blocks[0].values, np.full((4, 3, 1), 3.0))

        # block by kwargs
        blocks = tensor.blocks(key_1=1, key_2=0)
        assert len(blocks) == 1
        assert_equal(blocks[0].values, np.full((3, 1, 3), 2.0))

        # more than one block
        blocks = tensor.blocks(key_2=0)
        assert len(blocks) == 2

        assert_equal(blocks[0].values, np.full((3, 1, 1), 1.0))
        assert_equal(blocks[1].values, np.full((3, 1, 3), 2.0))

    def test_iter(self, tensor):
        expected = [
            ((0, 0), np.full((3, 1, 1), 1.0)),
            ((1, 0), np.full((3, 1, 3), 2.0)),
            ((2, 2), np.full((4, 3, 1), 3.0)),
            ((2, 3), np.full((4, 3, 1), 4.0)),
        ]
        for i, (sparse, block) in enumerate(tensor):
            expected_sparse, expected_values = expected[i]

            assert tuple(sparse) == expected_sparse
            assert_equal(block.values, expected_values)

    def test_keys_to_properties(self, tensor):
        tensor = tensor.keys_to_properties("key_1")

        assert tensor.keys.names == ("key_2",)
        assert tuple(tensor.keys[0]) == (0,)
        assert tuple(tensor.keys[1]) == (2,)
        assert tuple(tensor.keys[2]) == (3,)

        # The new first block contains the old first two blocks merged
        block = tensor.block(0)
        assert tuple(block.samples[0]) == (0,)
        assert tuple(block.samples[1]) == (1,)
        assert tuple(block.samples[2]) == (2,)
        assert tuple(block.samples[3]) == (3,)
        assert tuple(block.samples[4]) == (4,)

        assert len(block.components), 1
        assert tuple(block.components[0][0]), (0,)

        assert block.properties.names == ("key_1", "properties")
        assert tuple(block.properties[0]) == (0, 0)
        assert tuple(block.properties[1]) == (1, 3)
        assert tuple(block.properties[2]) == (1, 4)
        assert tuple(block.properties[3]) == (1, 5)

        expected = np.array(
            [
                [[1.0, 2.0, 2.0, 2.0]],
                [[0.0, 2.0, 2.0, 2.0]],
                [[1.0, 0.0, 0.0, 0.0]],
                [[0.0, 2.0, 2.0, 2.0]],
                [[1.0, 0.0, 0.0, 0.0]],
            ]
        )
        assert_equal(block.values, expected)

        gradient = block.gradient("parameter")
        assert tuple(gradient.samples[0]) == (0, -2)
        assert tuple(gradient.samples[1]) == (0, 3)
        assert tuple(gradient.samples[2]) == (3, -2)
        assert tuple(gradient.samples[3]) == (4, 3)

        expected = np.array(
            [
                [[11.0, 12.0, 12.0, 12.0]],
                [[0.0, 12.0, 12.0, 12.0]],
                [[0.0, 12.0, 12.0, 12.0]],
                [[11.0, 0.0, 0.0, 0.0]],
            ]
        )
        assert_equal(gradient.data, expected)

        # The new second block contains the old third block
        block = tensor.block(1)
        assert block.properties.names == ("key_1", "properties")
        assert tuple(block.properties[0]) == (2, 0)

        assert_equal(block.values, np.full((4, 3, 1), 3.0))

        # The new third block contains the old fourth block
        block = tensor.block(2)
        assert block.properties.names == ("key_1", "properties")
        assert tuple(block.properties[0]) == (2, 0)

        assert_equal(block.values, np.full((4, 3, 1), 4.0))

    def test_keys_to_samples(self, tensor):
        tensor = tensor_map().keys_to_samples("key_2", sort_samples=True)

        assert tensor.keys.names == ("key_1",)
        assert tuple(tensor.keys[0]) == (0,)
        assert tuple(tensor.keys[1]) == (1,)
        assert tuple(tensor.keys[2]) == (2,)

        # The first two blocks are not modified
        block = tensor.block(0)
        assert block.samples.names, ("samples", "key_2")
        assert tuple(block.samples[0]) == (0, 0)
        assert tuple(block.samples[1]) == (2, 0)
        assert tuple(block.samples[2]) == (4, 0)

        assert_equal(block.values, np.full((3, 1, 1), 1.0))

        block = tensor.block(1)
        assert block.samples.names == ("samples", "key_2")
        assert tuple(block.samples[0]) == (0, 0)
        assert tuple(block.samples[1]) == (1, 0)
        assert tuple(block.samples[2]) == (3, 0)

        assert_equal(block.values, np.full((3, 1, 3), 2.0))

        # The new third block contains the old third and fourth blocks merged
        block = tensor.block(2)

        assert block.samples.names == ("samples", "key_2")
        assert tuple(block.samples[0]) == (0, 2)
        assert tuple(block.samples[1]) == (0, 3)
        assert tuple(block.samples[2]) == (1, 3)
        assert tuple(block.samples[3]) == (2, 3)
        assert tuple(block.samples[4]) == (3, 2)
        assert tuple(block.samples[5]) == (5, 3)
        assert tuple(block.samples[6]) == (6, 2)
        assert tuple(block.samples[7]) == (8, 2)

        expected = np.array(
            [
                [[3.0], [3.0], [3.0]],
                [[4.0], [4.0], [4.0]],
                [[4.0], [4.0], [4.0]],
                [[4.0], [4.0], [4.0]],
                [[3.0], [3.0], [3.0]],
                [[4.0], [4.0], [4.0]],
                [[3.0], [3.0], [3.0]],
                [[3.0], [3.0], [3.0]],
            ]
        )
        assert_equal(block.values, expected)

        gradient = block.gradient("parameter")
        assert gradient.samples.names == ("sample", "parameter")
        assert tuple(gradient.samples[0]) == (1, 1)
        assert tuple(gradient.samples[1]) == (4, -2)
        assert tuple(gradient.samples[2]) == (5, 3)

        expected = np.array(
            [
                [[14.0], [14.0], [14.0]],
                [[13.0], [13.0], [13.0]],
                [[14.0], [14.0], [14.0]],
            ]
        )
        assert_equal(gradient.data, expected)

    def test_keys_to_samples_unsorted(self, tensor):
        tensor = tensor.keys_to_samples("key_2", sort_samples=False)

        block = tensor.block(2)
        assert block.samples.names, ("samples", "key_2")
        assert tuple(block.samples[0]) == (0, 2)
        assert tuple(block.samples[1]) == (3, 2)
        assert tuple(block.samples[2]) == (6, 2)
        assert tuple(block.samples[3]) == (8, 2)
        assert tuple(block.samples[4]) == (0, 3)
        assert tuple(block.samples[5]) == (1, 3)
        assert tuple(block.samples[6]) == (2, 3)
        assert tuple(block.samples[7]) == (5, 3)

    def test_components_to_properties(self, tensor):
        tensor = tensor.components_to_properties("components")

        block = tensor.block(0)
        assert block.samples.names == ("samples",)
        assert tuple(block.samples[0]) == (0,)
        assert tuple(block.samples[1]) == (2,)
        assert tuple(block.samples[2]) == (4,)

        assert block.components == []

        assert block.properties.names == ("components", "properties")
        assert tuple(block.properties[0]) == (0, 0)

        block = tensor.block(3)
        assert block.samples.names, ("samples",)
        assert tuple(block.samples[0]) == (0,)
        assert tuple(block.samples[1]) == (1,)
        assert tuple(block.samples[2]) == (2,)
        assert tuple(block.samples[3]) == (5,)

        assert block.components == []

        assert block.properties.names == ("components", "properties")
        assert tuple(block.properties[0]) == (0, 0)
        assert tuple(block.properties[1]) == (1, 0)
        assert tuple(block.properties[2]) == (2, 0)

    def test_eq(self, tensor):
        assert equistore.equal(tensor, tensor) == (tensor == tensor)

    def test_neq(self, tensor, large_tensor):
        assert equistore.equal(tensor, large_tensor) == (tensor == large_tensor)

    def test_add(self, tensor):
        assert equistore.add(tensor, 1) == (tensor + 1)

    def test_sub(self, tensor):
        assert equistore.subtract(tensor, 1) == (tensor - 1)

    def test_mul(self, tensor):
        assert equistore.multiply(tensor, 2) == (tensor * 2)

    def test_matmul(self, tensor):
        tensor = tensor.components_to_properties("components")
        tensor = equistore.remove_gradients(tensor)

        assert equistore.dot(tensor, tensor) == (tensor @ tensor)

    def test_truediv(self, tensor):
        assert equistore.divide(tensor, 2) == (tensor / 2)

    def test_pow(self, tensor):
        assert equistore.pow(tensor, 2) == (tensor**2)
