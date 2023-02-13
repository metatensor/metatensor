import os
import unittest

import numpy as np

import equistore.io
import equistore.operations as fn
from equistore import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


class TestEqual(unittest.TestCase):
    def test_equal_nograd(self):
        block_1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]], dtype=np.float64),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_2 = TensorBlock(
            values=np.array([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]]),
            samples=Labels(
                ["samples"],
                np.array([[0], [1], [2], [3], [4], [5]], dtype=np.int32),
            ),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )

        block_3 = TensorBlock(
            values=np.array([[1], [2]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )
        block_4 = TensorBlock(
            values=np.array([[23], [53], [83]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[6]], dtype=np.int32)),
        )
        keys = Labels(
            names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]], dtype=np.int32)
        )
        X = TensorMap(keys, [block_1, block_2])
        self.assertTrue(fn.equal(X, X))
        Y = TensorMap(keys, [block_3, block_4])
        self.assertFalse(fn.equal(X, Y))
        with self.assertRaises(ValueError) as cm:
            fn.equal_raise(X, Y)

        self.assertEqual(
            str(cm.exception), "The TensorBlocks with key = (0, 0) are different"
        )

        block_1_c = TensorBlock(
            values=np.array(
                [
                    [
                        [[1, 0.5], [4, 2], [1.5, 6.5]],
                        [[2, 1], [6, 3], [6.1, 3.5]],
                        [[9, 9], [9, 9.8], [10, 10.5]],
                    ],
                    [
                        [[3, 1.5], [7, 3.5], [3.7, 1.5]],
                        [[5, 2.5], [8, 4], [6.3, 1.5]],
                        [[5, 7.1], [8, 4.8], [6.3, 14.466]],
                    ],
                ],
                dtype=np.float64,
            ),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[
                Labels(["c1"], np.array([[0], [1], [2]], dtype=np.int32)),
                Labels(["c2"], np.array([[0], [1], [2]], dtype=np.int32)),
            ],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_1_c_copy = TensorBlock(
            values=block_1_c.values + 0.1e-6,
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[
                Labels(["c1"], np.array([[0], [1], [2]], dtype=np.int32)),
                Labels(["c2"], np.array([[0], [1], [2]], dtype=np.int32)),
            ],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )

        block_2_c = TensorBlock(
            values=np.array(
                [
                    [[[1, 2], [6.8, 2.8]], [[4.1, 6.2], [66.8, 62.8]]],
                    [[[3, 4], [36, 6.4]], [[83, 84], [73.76, 76.74]]],
                    [[[5, 6], [58, 68]], [[23.4, 5643.3], [234.5, 3247.6]]],
                    [[[5.6, 6.6], [5.68, 668]], [[55.6, 676.76], [775.68, 0.668]]],
                    [[[1, 2], [17.7, 27.7]], [[77.1, 22.2], [1.11, 3.42]]],
                ]
            ),
            samples=Labels(
                ["samples"],
                np.array([[0], [1], [2], [3], [4]], dtype=np.int32),
            ),
            components=[
                Labels(["c1"], np.array([[3], [5]], dtype=np.int32)),
                Labels(["c2"], np.array([[6], [8]], dtype=np.int32)),
            ],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_2_c_copy = TensorBlock(
            values=block_2_c.values + 0.1e-6,
            samples=Labels(
                ["samples"],
                np.array([[0], [1], [2], [3], [4]], dtype=np.int32),
            ),
            components=[
                Labels(["c1"], np.array([[3], [5]], dtype=np.int32)),
                Labels(["c2"], np.array([[6], [8]], dtype=np.int32)),
            ],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        X_c = TensorMap(keys, [block_1_c, block_2_c])
        X_c_copy = TensorMap(keys, [block_1_c_copy, block_2_c_copy])
        self.assertFalse(fn.equal(X, X_c))

        self.assertTrue(fn.equal(X_c, X_c))
        self.assertFalse(fn.equal(X_c, X_c_copy))

    def test_self_equal_grad(self):
        tensor1 = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            use_numpy=True,
        )
        blocks = []
        blocks_e6 = []
        for _, block in tensor1:
            blocks.append(block.copy())
            be6 = block.copy()
            be6.values[:] += 1e-6
            blocks_e6.append(be6)

        tensor1_copy = TensorMap(tensor1.keys, blocks)
        tensor1_e6 = TensorMap(tensor1.keys, blocks_e6)
        self.assertTrue(fn.equal(tensor1, tensor1_copy))
        self.assertFalse(fn.equal(tensor1, tensor1_e6))

        with self.assertRaises(ValueError) as cm:
            fn.equal_raise(tensor1, tensor1_e6)

        self.assertEqual(
            str(cm.exception), "The TensorBlocks with key = (0, 1, 1) are different"
        )

        with self.assertRaises(ValueError) as cm:
            fn.equal_block_raise(tensor1.block(0), tensor1_e6.block(0))
        self.assertEqual(str(cm.exception), "values are not equal")

    def test_self_equal_exceptions(self):
        block_1 = TensorBlock(
            values=np.array([[1], [2]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )
        block_2 = TensorBlock(
            values=np.array([[1], [2]]),
            samples=Labels(["samples_5"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )
        block_3 = TensorBlock(
            values=np.array([[1], [2]]),
            samples=Labels(["samples"], np.array([[0], [6]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )

        block_4 = TensorBlock(
            values=np.array([[[1], [4]], [[44], [2]]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[
                Labels(["component"], np.array([[0], [6]], dtype=np.int32)),
            ],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )
        block_5 = TensorBlock(
            values=np.array([[[1], [4]], [[44], [2]]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[
                Labels(["component1"], np.array([[0], [6]], dtype=np.int32)),
            ],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )

        block_6 = TensorBlock(
            values=np.array([[[1], [4]], [[44], [2]]]),
            samples=Labels(["samples"], np.array([[2], [0]], dtype=np.int32)),
            components=[
                Labels(["component"], np.array([[0], [6]], dtype=np.int32)),
            ],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )

        self.assertFalse(fn.equal_block(block_1, block_2))

        with self.assertRaises(ValueError) as cm:
            fn.equal_block_raise(block_1, block_2)

        self.assertEqual(
            str(cm.exception),
            "Inputs to 'equal' should have the same samples:\n"
            "samples names are not the same or not in the same order.",
        )

        with self.assertRaises(ValueError) as cm:
            fn.equal_block_raise(block_1, block_3)

        self.assertEqual(
            str(cm.exception),
            "Inputs to 'equal' should have the same samples:\n"
            "samples are not the same or not in the same order.",
        )

        with self.assertRaises(ValueError) as cm:
            fn.equal_block_raise(block_1, block_4)

        self.assertEqual(str(cm.exception), "values shapes are different")

        with self.assertRaises(ValueError) as cm:
            fn.equal_block_raise(block_5, block_4)

        self.assertEqual(
            str(cm.exception),
            "Inputs to 'equal' should have the same components:\n"
            "components names are not the same or not in the same order.",
        )

        with self.assertRaises(ValueError) as cm:
            fn.equal_block_raise(block_6, block_4)

        self.assertEqual(
            str(cm.exception),
            "Inputs to 'equal' should have the same samples:\n"
            "samples are not the same or not in the same order.",
        )

    def test_self_equal_exceptions_gradient(self):
        block_1 = TensorBlock(
            values=np.array([[1], [2]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )

        block_1.add_gradient(
            "parameter",
            data=np.full((2, 1), 11.0),
            samples=Labels(
                ["sample", "parameter"], np.array([[0, -2], [2, 3]], dtype=np.int32)
            ),
            components=[],
        )

        block_2 = TensorBlock(
            values=np.array([[1], [2]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )

        block_2.add_gradient(
            "parameter",
            data=np.full((2, 1), 11.0),
            samples=Labels(
                ["sample", "parameter1"], np.array([[0, -2], [2, 3]], dtype=np.int32)
            ),
            components=[],
        )

        with self.assertRaises(ValueError) as cm:
            fn.equal_block_raise(block_1, block_2)

        self.assertEqual(
            str(cm.exception),
            "Inputs to equal should have the same gradient:\n"
            'gradient ("parameter") samples names are not the same'
            " or not in the same order.",
        )

        block_3 = TensorBlock(
            values=np.array([[1], [2]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )

        block_3.add_gradient(
            "parameter",
            data=np.full((2, 1), 1.0),
            samples=Labels(
                ["sample", "parameter"], np.array([[0, -2], [2, 3]], dtype=np.int32)
            ),
            components=[],
        )

        with self.assertRaises(ValueError) as cm:
            fn.equal_block_raise(block_1, block_3)

        self.assertEqual(
            str(cm.exception),
            'gradient ("parameter") data are not equal',
        )

        block_4 = TensorBlock(
            values=np.array([[1], [2]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )

        block_4.add_gradient(
            "parameter",
            data=np.full((2, 3, 1), 1.0),
            samples=Labels(
                ["sample", "parameter"], np.array([[0, -2], [2, 3]], dtype=np.int32)
            ),
            components=[
                Labels(["component_1"], np.array([[-1], [0], [1]], dtype=np.int32))
            ],
        )

        block_5 = TensorBlock(
            values=np.array([[1], [2]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )

        block_5.add_gradient(
            "parameter",
            data=np.full((2, 3, 1), 1.0),
            samples=Labels(
                ["sample", "parameter"], np.array([[0, -2], [2, 3]], dtype=np.int32)
            ),
            components=[
                Labels(["component_1"], np.array([[-1], [6], [1]], dtype=np.int32))
            ],
        )

        with self.assertRaises(ValueError) as cm:
            fn.equal_block_raise(block_5, block_4)

        self.assertEqual(
            str(cm.exception),
            "Inputs to equal should have the same gradient:\n"
            'gradient ("parameter") components are not the same'
            " or not in the same order.",
        )

        block_6 = TensorBlock(
            values=np.array([[1], [2]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )

        block_6.add_gradient(
            "parameter",
            data=np.full((2, 3, 1), 1.0),
            samples=Labels(
                ["sample", "parameter_1"], np.array([[0, -2], [2, 3]], dtype=np.int32)
            ),
            components=[
                Labels(["component_1"], np.array([[-1], [6], [1]], dtype=np.int32))
            ],
        )
        with self.assertRaises(ValueError) as cm:
            fn.equal_block_raise(block_5, block_6)

        self.assertEqual(
            str(cm.exception),
            "Inputs to equal should have the same gradient:\n"
            'gradient ("parameter") samples names are not the same'
            " or not in the same order.",
        )


# TODO: add tests with torch & torch scripting/tracing

if __name__ == "__main__":
    unittest.main()
