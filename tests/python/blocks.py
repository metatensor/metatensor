import unittest
import numpy as np


from aml_storage import Block, Labels
from utils import test_descriptor


class TestBlocks(unittest.TestCase):
    def test_create_block(self):
        block = Block(
            values=np.full((3, 1, 1), -1.0),
            samples=Labels(["samples"], np.array([[0], [2], [4]], dtype=np.int32)),
            components=Labels(["components"], np.array([[-2]], dtype=np.int32)),
            features=Labels(["features"], np.array([[5]], dtype=np.int32)),
        )

        self.assertTrue(np.all(block.values == np.full((3, 1, 1), -1.0)))

        self.assertEqual(block.samples.names, ("samples",))
        self.assertEqual(len(block.samples), 3)
        self.assertEqual(tuple(block.samples[0]), (0,))
        self.assertEqual(tuple(block.samples[1]), (2,))
        self.assertEqual(tuple(block.samples[2]), (4,))

        self.assertEqual(block.components.names, ("components",))
        self.assertEqual(len(block.components), 1)
        self.assertEqual(tuple(block.components[0]), (-2,))

        self.assertEqual(block.features.names, ("features",))
        self.assertEqual(len(block.features), 1)
        self.assertEqual(tuple(block.features[0]), (5,))

    def test_gradients(self):
        block = test_descriptor().block(0)

        self.assertTrue(block.has_gradient("parameter"))
        self.assertFalse(block.has_gradient("something else"))

        samples, gradients = block.gradient("parameter")

        self.assertEqual(samples.names, ("sample", "parameter"))
        self.assertEqual(len(samples), 2)
        self.assertEqual(tuple(samples[0]), (0, -2))
        self.assertEqual(tuple(samples[1]), (2, 3))

        self.assertTrue(np.all(gradients == np.full((2, 1, 1), 11.0)))


if __name__ == "__main__":
    unittest.main()
