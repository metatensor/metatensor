import unittest
import numpy as np

from utils import test_descriptor


class TestDescriptors(unittest.TestCase):
    def test_sparse(self):
        descriptor = test_descriptor()
        self.assertEqual(descriptor.sparse.names, ("sparse_1", "sparse_2"))
        self.assertEqual(len(descriptor.sparse), 4)
        self.assertEqual(tuple(descriptor.sparse[0]), (0, 0))
        self.assertEqual(tuple(descriptor.sparse[1]), (1, 0))
        self.assertEqual(tuple(descriptor.sparse[2]), (2, 2))
        self.assertEqual(tuple(descriptor.sparse[3]), (2, 3))

    def test_labels_names(self):
        descriptor = test_descriptor()

        self.assertEqual(descriptor.sample_names, ("samples",))
        self.assertEqual(descriptor.component_names, [("components",)])
        self.assertEqual(descriptor.feature_names, ("features",))

    def test_get_block(self):
        descriptor = test_descriptor()

        # block by index
        block = descriptor.block(2)
        self.assertTrue(np.all(block.values == np.full((4, 3, 1), 3.0)))

        # block by kwargs
        block = descriptor.block(sparse_1=1, sparse_2=0)
        self.assertTrue(np.all(block.values == np.full((3, 1, 1), 2.0)))

        # block by Label entry
        block = descriptor.block(descriptor.sparse[0])
        self.assertTrue(np.all(block.values == np.full((3, 1, 1), 1.0)))

    def test_iter(self):
        expected = [
            ((0, 0), np.full((3, 1, 1), 1.0)),
            ((1, 0), np.full((3, 1, 1), 2.0)),
            ((2, 2), np.full((4, 3, 1), 3.0)),
            ((2, 3), np.full((4, 3, 1), 4.0)),
        ]

        descriptor = test_descriptor()
        for i, (sparse, block) in enumerate(descriptor):
            expected_sparse, expected_values = expected[i]

            self.assertEqual(tuple(sparse), expected_sparse)
            self.assertTrue(np.all(block.values == expected_values))

    def test_sparse_to_features(self):
        pass

    def test_sparse_to_samples(self):
        pass

    def test_components_to_features(self):
        pass


if __name__ == "__main__":
    unittest.main()
