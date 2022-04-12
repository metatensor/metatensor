import unittest
import numpy as np


from aml_storage import Block, Labels


class TestBlocks(unittest.TestCase):
    def test_block_no_components(self):
        block = Block(
            values=np.full((3, 2), -1.0),
            samples=Labels(["samples"], np.array([[0], [2], [4]], dtype=np.int32)),
            components=[],
            features=Labels(["features"], np.array([[5], [3]], dtype=np.int32)),
        )

        self.assertTrue(np.all(block.values == np.full((3, 2), -1.0)))

        self.assertEqual(block.samples.names, ("samples",))
        self.assertEqual(len(block.samples), 3)
        self.assertEqual(tuple(block.samples[0]), (0,))
        self.assertEqual(tuple(block.samples[1]), (2,))
        self.assertEqual(tuple(block.samples[2]), (4,))

        self.assertEqual(len(block.components), 0)

        self.assertEqual(block.features.names, ("features",))
        self.assertEqual(len(block.features), 2)
        self.assertEqual(tuple(block.features[0]), (5,))
        self.assertEqual(tuple(block.features[1]), (3,))

    def test_block_with_components(self):
        block = Block(
            values=np.full((3, 3, 2, 2), -1.0),
            samples=Labels(["samples"], np.array([[0], [2], [4]], dtype=np.int32)),
            components=[
                Labels(["component_1"], np.array([[-1], [0], [1]], dtype=np.int32)),
                Labels(["component_2"], np.array([[-4], [1]], dtype=np.int32)),
            ],
            features=Labels(["features"], np.array([[5], [3]], dtype=np.int32)),
        )

        self.assertTrue(np.all(block.values == np.full((3, 3, 2, 2), -1.0)))

        self.assertEqual(block.samples.names, ("samples",))
        self.assertEqual(len(block.samples), 3)
        self.assertEqual(tuple(block.samples[0]), (0,))
        self.assertEqual(tuple(block.samples[1]), (2,))
        self.assertEqual(tuple(block.samples[2]), (4,))

        self.assertEqual(len(block.components), 2)
        component_1 = block.components[0]
        self.assertEqual(component_1.names, ("component_1",))
        self.assertEqual(len(component_1), 3)
        self.assertEqual(tuple(component_1[0]), (-1,))
        self.assertEqual(tuple(component_1[1]), (0,))
        self.assertEqual(tuple(component_1[2]), (1,))

        component_2 = block.components[1]
        self.assertEqual(component_2.names, ("component_2",))
        self.assertEqual(len(component_2), 2)
        self.assertEqual(tuple(component_2[0]), (-4,))
        self.assertEqual(tuple(component_2[1]), (1,))

        self.assertEqual(block.features.names, ("features",))
        self.assertEqual(len(block.features), 2)
        self.assertEqual(tuple(block.features[0]), (5,))
        self.assertEqual(tuple(block.features[1]), (3,))

    def test_gradients(self):
        block = Block(
            values=np.full((3, 3, 2, 2), -1.0),
            samples=Labels(["samples"], np.array([[0], [2], [4]], dtype=np.int32)),
            components=[
                Labels(["component_1"], np.array([[-1], [0], [1]], dtype=np.int32)),
                Labels(["component_2"], np.array([[-4], [1]], dtype=np.int32)),
            ],
            features=Labels(["features"], np.array([[5], [3]], dtype=np.int32)),
        )

        block.add_gradient(
            "parameter",
            data=np.full((2, 3, 2, 2), 11.0),
            samples=Labels(
                ["sample", "parameter"], np.array([[0, -2], [2, 3]], dtype=np.int32)
            ),
            components=[
                Labels(["component_1"], np.array([[-1], [0], [1]], dtype=np.int32)),
                Labels(["component_2"], np.array([[-4], [1]], dtype=np.int32)),
            ],
        )

        self.assertTrue(block.has_gradient("parameter"))
        self.assertFalse(block.has_gradient("something else"))

        self.assertEqual(block.gradients_list(), ["parameter"])

        gradient = block.gradient("parameter")

        self.assertEqual(gradient.samples.names, ("sample", "parameter"))
        self.assertEqual(len(gradient.samples), 2)
        self.assertEqual(tuple(gradient.samples[0]), (0, -2))
        self.assertEqual(tuple(gradient.samples[1]), (2, 3))

        self.assertTrue(np.all(gradient.data == np.full((2, 3, 2, 2), 11.0)))

    def test_copy(self):
        block = Block(
            values=np.full((3, 3, 2), 2.0),
            samples=Labels(["samples"], np.array([[0], [2], [4]], dtype=np.int32)),
            components=[
                Labels(["component_1"], np.array([[-1], [0], [1]], dtype=np.int32)),
            ],
            features=Labels(["features"], np.array([[5], [3]], dtype=np.int32)),
        )
        copy = block.copy()
        block_values_id = id(block.values)

        del block

        self.assertNotEqual(id(copy.values), block_values_id)

        self.assertTrue(np.all(copy.values == np.full((3, 3, 2), 2.0)))
        self.assertEqual(copy.samples.names, ("samples",))
        self.assertEqual(len(copy.samples), 3)
        self.assertEqual(tuple(copy.samples[0]), (0,))
        self.assertEqual(tuple(copy.samples[1]), (2,))
        self.assertEqual(tuple(copy.samples[2]), (4,))


if __name__ == "__main__":
    unittest.main()
