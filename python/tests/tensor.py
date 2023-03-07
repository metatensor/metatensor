import unittest

import numpy as np
from utils import test_large_tensor_map, test_tensor_map

import equistore


class TestTensorMap(unittest.TestCase):
    def setUp(self):
        self.tm = test_tensor_map()

    def test_keys(self):
        self.assertEqual(self.tm.keys.names, ("key_1", "key_2"))
        self.assertEqual(len(self.tm.keys), 4)
        self.assertEqual(len(self.tm), 4)
        self.assertEqual(tuple(self.tm.keys[0]), (0, 0))
        self.assertEqual(tuple(self.tm.keys[1]), (1, 0))
        self.assertEqual(tuple(self.tm.keys[2]), (2, 2))
        self.assertEqual(tuple(self.tm.keys[3]), (2, 3))

    def test_print(self):
        """
        Test routine for the print function of the TensorBlock.
        It compare the results with those in a file.
        """
        repr = self.tm.__repr__()
        expected = """TensorMap with 4 blocks
keys: ['key_1' 'key_2']
          0       0
          1       0
          2       2
          2       3"""
        self.assertTrue(expected == repr)

        tm = test_large_tensor_map()
        _print = tm.__repr__()
        expected = """TensorMap with 12 blocks
keys: ['key_1' 'key_2']
          0       0
          1       0
          2       2
       ...
          1       5
          2       5
          3       5"""
        self.assertTrue(expected == _print)

    def test_labels_names(self):
        self.assertEqual(self.tm.sample_names, ("samples",))
        self.assertEqual(self.tm.components_names, [("components",)])
        self.assertEqual(self.tm.property_names, ("properties",))

    def test_block(self):
        # block by index
        block = self.tm.block(2)
        self.assertTrue(np.all(block.values == np.full((4, 3, 1), 3.0)))

        # block by index with __getitem__
        block = self.tm[2]
        self.assertTrue(np.all(block.values == np.full((4, 3, 1), 3.0)))

        # block by kwargs
        block = self.tm.block(key_1=1, key_2=0)
        self.assertTrue(np.all(block.values == np.full((3, 1, 1), 2.0)))

        # block by Label entry
        block = self.tm.block(self.tm.keys[0])
        self.assertTrue(np.all(block.values == np.full((3, 1, 1), 1.0)))

        # block by Label entry with __getitem__
        block = self.tm[self.tm.keys[0]]
        self.assertTrue(np.all(block.values == np.full((3, 1, 1), 1.0)))

        # More arguments than needed: two integers
        # by index
        with self.assertRaises(ValueError) as cm:
            self.tm.block(3, 4)

        self.assertEqual(
            str(cm.exception),
            "only one non-keyword argument is supported, 2 are given",
        )

        # 4 input with the first as integer by __getitem__
        with self.assertRaises(ValueError) as cm:
            self.tm[3, 4, 7.0, "r"]

        self.assertEqual(
            str(cm.exception),
            "only one non-keyword argument is supported, 4 are given",
        )

        # More arguments than needed: 3 Labels
        with self.assertRaises(ValueError) as cm:
            self.tm.block(self.tm.keys[0], self.tm.keys[1], self.tm.keys[3])

        self.assertEqual(
            str(cm.exception),
            "only one non-keyword argument is supported, 3 are given",
        )

        # by __getitem__
        with self.assertRaises(ValueError) as cm:
            self.tm[self.tm.keys[1], 4]

        self.assertEqual(
            str(cm.exception),
            "only one non-keyword argument is supported, 2 are given",
        )

        # 0 blocks matching criteria
        with self.assertRaises(ValueError) as cm:
            self.tm.block(key_1=3)

        self.assertEqual(
            str(cm.exception),
            "Couldn't find any block matching the selection 'key_1 = 3'",
        )

        # more than one block matching criteria
        with self.assertRaises(ValueError) as cm:
            self.tm.block(key_2=0)

        self.assertEqual(
            str(cm.exception),
            "more than one block matched 'key_2 = 0', use `TensorMap.blocks` "
            "if you want to get all of them",
        )

    def test_blocks(self):
        # block by index
        blocks = self.tm.blocks(2)
        self.assertEqual(len(blocks), 1)
        self.assertTrue(np.all(blocks[0].values == np.full((4, 3, 1), 3.0)))

        # block by kwargs
        blocks = self.tm.blocks(key_1=1, key_2=0)
        self.assertEqual(len(blocks), 1)
        self.assertTrue(np.all(blocks[0].values == np.full((3, 1, 1), 2.0)))

        # more than one block
        blocks = self.tm.blocks(key_2=0)
        self.assertEqual(len(blocks), 2)

        self.assertTrue(np.all(blocks[0].values == np.full((3, 1, 1), 1.0)))
        self.assertTrue(np.all(blocks[1].values == np.full((3, 1, 3), 2.0)))

    def test_iter(self):
        expected = [
            ((0, 0), np.full((3, 1, 1), 1.0)),
            ((1, 0), np.full((3, 1, 1), 2.0)),
            ((2, 2), np.full((4, 3, 1), 3.0)),
            ((2, 3), np.full((4, 3, 1), 4.0)),
        ]

        for i, (sparse, block) in enumerate(self.tm):
            expected_sparse, expected_values = expected[i]

            self.assertEqual(tuple(sparse), expected_sparse)
            self.assertTrue(np.all(block.values == expected_values))

    def test_keys_to_properties(self):
        self.tm = self.tm.keys_to_properties("key_1")

        self.assertEqual(self.tm.keys.names, ("key_2",))
        self.assertEqual(tuple(self.tm.keys[0]), (0,))
        self.assertEqual(tuple(self.tm.keys[1]), (2,))
        self.assertEqual(tuple(self.tm.keys[2]), (3,))

        # The new first block contains the old first two blocks merged
        block = self.tm.block(0)
        self.assertEqual(tuple(block.samples[0]), (0,))
        self.assertEqual(tuple(block.samples[1]), (1,))
        self.assertEqual(tuple(block.samples[2]), (2,))
        self.assertEqual(tuple(block.samples[3]), (3,))
        self.assertEqual(tuple(block.samples[4]), (4,))

        self.assertEqual(len(block.components), 1)
        self.assertEqual(tuple(block.components[0][0]), (0,))

        self.assertEqual(block.properties.names, ("key_1", "properties"))
        self.assertEqual(tuple(block.properties[0]), (0, 0))
        self.assertEqual(tuple(block.properties[1]), (1, 3))
        self.assertEqual(tuple(block.properties[2]), (1, 4))
        self.assertEqual(tuple(block.properties[3]), (1, 5))

        expected = np.array(
            [
                [[1.0, 2.0, 2.0, 2.0]],
                [[0.0, 2.0, 2.0, 2.0]],
                [[1.0, 0.0, 0.0, 0.0]],
                [[0.0, 2.0, 2.0, 2.0]],
                [[1.0, 0.0, 0.0, 0.0]],
            ]
        )
        self.assertTrue(np.all(block.values == expected))

        gradient = block.gradient("parameter")
        self.assertEqual(tuple(gradient.samples[0]), (0, -2))
        self.assertEqual(tuple(gradient.samples[1]), (0, 3))
        self.assertEqual(tuple(gradient.samples[2]), (3, -2))
        self.assertEqual(tuple(gradient.samples[3]), (4, 3))

        expected = np.array(
            [
                [[11.0, 12.0, 12.0, 12.0]],
                [[0.0, 12.0, 12.0, 12.0]],
                [[0.0, 12.0, 12.0, 12.0]],
                [[11.0, 0.0, 0.0, 0.0]],
            ]
        )
        self.assertTrue(np.all(gradient.data == expected))

        # The new second block contains the old third block
        block = self.tm.block(1)
        self.assertEqual(block.properties.names, ("key_1", "properties"))
        self.assertEqual(tuple(block.properties[0]), (2, 0))

        self.assertTrue(np.all(block.values == np.full((4, 3, 1), 3.0)))

        # The new third block contains the old fourth block
        block = self.tm.block(2)
        self.assertEqual(block.properties.names, ("key_1", "properties"))
        self.assertEqual(tuple(block.properties[0]), (2, 0))

        self.assertTrue(np.all(block.values == np.full((4, 3, 1), 4.0)))

    def test_keys_to_samples(self):
        self.tm = self.tm.keys_to_samples("key_2", sort_samples=True)

        self.assertEqual(self.tm.keys.names, ("key_1",))
        self.assertEqual(tuple(self.tm.keys[0]), (0,))
        self.assertEqual(tuple(self.tm.keys[1]), (1,))
        self.assertEqual(tuple(self.tm.keys[2]), (2,))

        # The first two blocks are not modified
        block = self.tm.block(0)
        self.assertEqual(block.samples.names, ("samples", "key_2"))
        self.assertEqual(tuple(block.samples[0]), (0, 0))
        self.assertEqual(tuple(block.samples[1]), (2, 0))
        self.assertEqual(tuple(block.samples[2]), (4, 0))

        self.assertTrue(np.all(block.values == np.full((3, 1, 1), 1.0)))

        block = self.tm.block(1)
        self.assertEqual(block.samples.names, ("samples", "key_2"))
        self.assertEqual(tuple(block.samples[0]), (0, 0))
        self.assertEqual(tuple(block.samples[1]), (1, 0))
        self.assertEqual(tuple(block.samples[2]), (3, 0))

        self.assertTrue(np.all(block.values == np.full((3, 1, 3), 2.0)))

        # The new third block contains the old third and fourth blocks merged
        block = self.tm.block(2)

        self.assertEqual(block.samples.names, ("samples", "key_2"))
        self.assertEqual(tuple(block.samples[0]), (0, 2))
        self.assertEqual(tuple(block.samples[1]), (0, 3))
        self.assertEqual(tuple(block.samples[2]), (1, 3))
        self.assertEqual(tuple(block.samples[3]), (2, 3))
        self.assertEqual(tuple(block.samples[4]), (3, 2))
        self.assertEqual(tuple(block.samples[5]), (5, 3))
        self.assertEqual(tuple(block.samples[6]), (6, 2))
        self.assertEqual(tuple(block.samples[7]), (8, 2))

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
        self.assertTrue(np.all(block.values == expected))

        gradient = block.gradient("parameter")
        self.assertEqual(gradient.samples.names, ("sample", "parameter"))
        self.assertEqual(tuple(gradient.samples[0]), (1, 1))
        self.assertEqual(tuple(gradient.samples[1]), (4, -2))
        self.assertEqual(tuple(gradient.samples[2]), (5, 3))

        expected = np.array(
            [
                [[14.0], [14.0], [14.0]],
                [[13.0], [13.0], [13.0]],
                [[14.0], [14.0], [14.0]],
            ]
        )
        self.assertTrue(np.all(gradient.data == expected))

    def test_keys_to_samples_unsorted(self):
        tensor = self.tm.keys_to_samples("key_2", sort_samples=False)

        block = tensor.block(2)
        self.assertEqual(block.samples.names, ("samples", "key_2"))
        self.assertEqual(tuple(block.samples[0]), (0, 2))
        self.assertEqual(tuple(block.samples[1]), (3, 2))
        self.assertEqual(tuple(block.samples[2]), (6, 2))
        self.assertEqual(tuple(block.samples[3]), (8, 2))
        self.assertEqual(tuple(block.samples[4]), (0, 3))
        self.assertEqual(tuple(block.samples[5]), (1, 3))
        self.assertEqual(tuple(block.samples[6]), (2, 3))
        self.assertEqual(tuple(block.samples[7]), (5, 3))

    def test_components_to_properties(self):
        tensor = self.tm.components_to_properties("components")

        block = tensor.block(0)
        self.assertEqual(block.samples.names, ("samples",))
        self.assertEqual(tuple(block.samples[0]), (0,))
        self.assertEqual(tuple(block.samples[1]), (2,))
        self.assertEqual(tuple(block.samples[2]), (4,))

        self.assertEqual(block.components, [])

        self.assertEqual(block.properties.names, ("components", "properties"))
        self.assertEqual(tuple(block.properties[0]), (0, 0))

        block = tensor.block(3)
        self.assertEqual(block.samples.names, ("samples",))
        self.assertEqual(tuple(block.samples[0]), (0,))
        self.assertEqual(tuple(block.samples[1]), (1,))
        self.assertEqual(tuple(block.samples[2]), (2,))
        self.assertEqual(tuple(block.samples[3]), (5,))

        self.assertEqual(block.components, [])

        self.assertEqual(block.properties.names, ("components", "properties"))
        self.assertEqual(tuple(block.properties[0]), (0, 0))
        self.assertEqual(tuple(block.properties[1]), (1, 0))
        self.assertEqual(tuple(block.properties[2]), (2, 0))

    def test_eq(self):
        self.assertTrue(self.tm == self.tm)

    def test_neq(self):
        self.assertFalse(self.tm != self.tm)

    def test_add(self):
        self.assertTrue(equistore.add(self.tm, 1) == self.tm + 1)

    def test_sub(self):
        self.assertTrue(equistore.subtract(self.tm, 1) == self.tm - 1)

    def test_mul(self):
        self.assertTrue(equistore.multiply(self.tm, 2) == self.tm * 2)

    def test_matmul(self):
        tensor = self.tm
        tensor = tensor.components_to_properties("components")
        tensor = equistore.remove_gradients(tensor)

        self.assertTrue(equistore.dot(tensor, tensor) == tensor @ tensor)

    def test_truediv(self):
        self.assertTrue(equistore.divide(self.tm, 2) == self.tm / 2)

    def test_pow(self):
        self.assertTrue(equistore.pow(self.tm, 2) == self.tm**2)

    def test_neg(self):
        self.assertTrue(equistore.multiply(self.tm, -1) == -self.tm)

    def test_pos(self):
        self.assertTrue(equistore.multiply(self.tm, +1) == +self.tm)


if __name__ == "__main__":
    unittest.main()
