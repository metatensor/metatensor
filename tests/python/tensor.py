import unittest
import numpy as np

from utils import test_tensor_map


class TestTensorMap(unittest.TestCase):
    def test_keys(self):
        tensor = test_tensor_map()
        self.assertEqual(tensor.keys.names, ("key_1", "key_2"))
        self.assertEqual(len(tensor.keys), 4)
        self.assertEqual(tuple(tensor.keys[0]), (0, 0))
        self.assertEqual(tuple(tensor.keys[1]), (1, 0))
        self.assertEqual(tuple(tensor.keys[2]), (2, 2))
        self.assertEqual(tuple(tensor.keys[3]), (2, 3))

    def test_labels_names(self):
        tensor = test_tensor_map()

        self.assertEqual(tensor.sample_names, ("samples",))
        self.assertEqual(tensor.component_names, [("components",)])
        self.assertEqual(tensor.property_names, ("properties",))

    def test_get_block(self):
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
