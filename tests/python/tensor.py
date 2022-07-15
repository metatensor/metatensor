import unittest

import numpy as np
from utils import test_large_tensor_map, test_tensor_map


class TestTensorMap(unittest.TestCase):
    def test_keys(self):
        tensor = test_tensor_map()
        self.assertEqual(tensor.keys.names, ("key_1", "key_2"))
        self.assertEqual(len(tensor.keys), 4)
        self.assertEqual(len(tensor), 4)
        self.assertEqual(tuple(tensor.keys[0]), (0, 0))
        self.assertEqual(tuple(tensor.keys[1]), (1, 0))
        self.assertEqual(tuple(tensor.keys[2]), (2, 2))
        self.assertEqual(tuple(tensor.keys[3]), (2, 3))

    def test_print(self):
        """
        Test routine for the print function of the TensorBlock.
        It compare the reults with those in a file.
        """
        tensor = test_tensor_map()
        repr = tensor.__repr__()
        expected = """TensorMap with 4 blocks
keys: ['key_1' 'key_2']
          0       0
          1       0
          2       2
          2       3"""
        self.assertTrue(expected == repr)

        tensor = test_large_tensor_map()
        _print = tensor.__repr__()
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
        tensor = test_tensor_map()

        self.assertEqual(tensor.sample_names, ("samples",))
        self.assertEqual(tensor.components_names, [("components",)])
        self.assertEqual(tensor.property_names, ("properties",))

    def test_block(self):
        tensor = test_tensor_map()

        # block by index
        block = tensor.block(2)
        self.assertTrue(np.all(block.values == np.full((4, 3, 1), 3.0)))

        # block by kwargs
        block = tensor.block(key_1=1, key_2=0)
        self.assertTrue(np.all(block.values == np.full((3, 1, 1), 2.0)))

        # block by Label entry
        block = tensor.block(tensor.keys[0])
        self.assertTrue(np.all(block.values == np.full((3, 1, 1), 1.0)))

        # 0 blocks matching criteria
        with self.assertRaises(ValueError) as cm:
            tensor.block(key_1=3)

        self.assertEqual(
            str(cm.exception),
            "Couldn't find any block matching the selection {'key_1': 3}",
        )

        # more than one block matching criteria
        with self.assertRaises(ValueError) as cm:
            tensor.block(key_2=0)

        self.assertEqual(
            str(cm.exception),
            "more than one block matched {'key_2': 0}, use `TensorMap.blocks` "
            "if you want to get all of them",
        )

    def test_blocks(self):
        tensor = test_tensor_map()

        # block by index
        blocks = tensor.blocks(2)
        self.assertEqual(len(blocks), 1)
        self.assertTrue(np.all(blocks[0].values == np.full((4, 3, 1), 3.0)))

        # block by kwargs
        blocks = tensor.blocks(key_1=1, key_2=0)
        self.assertEqual(len(blocks), 1)
        self.assertTrue(np.all(blocks[0].values == np.full((3, 1, 1), 2.0)))

        # more than one block
        blocks = tensor.blocks(key_2=0)
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

        tensor = test_tensor_map()
        for i, (sparse, block) in enumerate(tensor):
            expected_sparse, expected_values = expected[i]

            self.assertEqual(tuple(sparse), expected_sparse)
            self.assertTrue(np.all(block.values == expected_values))

    def test_keys_to_properties(self):
        pass

    def test_keys_to_samples(self):
        pass

    def test_components_to_properties(self):
        pass


if __name__ == "__main__":
    unittest.main()
