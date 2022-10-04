import os
import unittest

import numpy as np
from utils import compare_blocks, get_value_linear_solve

import equistore.io
import equistore.operations as fn
from equistore import Labels, TensorBlock, TensorMap

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestSolve(unittest.TestCase):
    def test_self_lstsq_nograd(self):
        block_1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_2 = TensorBlock(
            values=np.array([[1, 2], [3, 4], [5, 6]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]], dtype=np.int32)),
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
        Y = TensorMap(keys, [block_3, block_4])

        # Try to do with solve -> Raise Error
        with self.assertRaises(ValueError) as cm:
            w = fn.solve(X, Y)

        self.assertEqual(
            str(cm.exception),
            "the values in each TensorBlock of X should be a 2D-square array",
        )

        # solve with least square
        w = fn.lstsq(X, Y)

        self.assertTrue(len(w) == 2)
        self.assertTrue(np.all(w.keys == X.keys))
        self.assertTrue(
            np.allclose(w.block(0).values, np.array([-1.0, 1.0]), rtol=1e-13)
        )
        self.assertTrue(np.allclose(w.block(1).values, np.array([7, 8]), rtol=1e-7))
        for key, blockw in w:
            self.assertTrue(np.all(blockw.samples == Y.block(key).properties))
            self.assertTrue(np.all(blockw.properties == X.block(key).properties))

        Ydot = fn.dot(X, w)
        self.assertTrue(np.all(Ydot.keys == Y.keys))
        for key, expected_block in Ydot:
            comparing_dict = compare_blocks(expected_block, Y.block(key), rtol=1e-3)
            if not comparing_dict["general"]:
                print(str(comparing_dict))
            self.assertTrue(comparing_dict["general"])
            self.assertTrue(
                np.allclose(expected_block.values, Y.block(key).values, rtol=1e-13)
            )

    def test_self_solve_nograd(self):
        block_1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_2 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
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
            values=np.array([[2], [4]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )
        keys = Labels(
            names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]], dtype=np.int32)
        )
        X = TensorMap(keys, [block_1, block_2])
        Y = TensorMap(keys, [block_3, block_4])
        w = fn.solve(X, Y)

        self.assertTrue(len(w) == 2)
        self.assertTrue(np.all(w.keys == X.keys))
        self.assertTrue(
            np.allclose(w.block(0).values, np.array([-1.0, 1.0]), rtol=1e-13)
        )
        self.assertTrue(
            np.allclose(w.block(1).values, np.array([-2.0, 2.0]), rtol=1e-7)
        )
        for key, blockw in w:
            self.assertTrue(np.all(blockw.samples == Y.block(key).properties))
            self.assertTrue(np.all(blockw.properties == X.block(key).properties))

        Ydot = fn.dot(X, w)
        self.assertTrue(np.all(Ydot.keys == Y.keys))
        for key, expected_block in Ydot:
            comparing_dict = compare_blocks(expected_block, Y.block(key), rtol=1e-3)
            if not comparing_dict["general"]:
                print(str(comparing_dict))
            self.assertTrue(comparing_dict["general"])
            self.assertTrue(
                np.allclose(expected_block.values, Y.block(key).values, rtol=1e-13)
            )

    def test_self_lstsq_grad(self):
        Xval, Xgradval, Yval, Ygradval = get_value_linear_solve()
        block_X = TensorBlock(
            values=Xval,
            samples=Labels(
                ["samples"], np.array([[0], [1], [2], [3], [4]], dtype=np.int32)
            ),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_X.add_gradient(
            "positions",
            data=Xgradval,
            samples=Labels(
                ["sample", "positions"],
                np.array([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]], dtype=np.int32),
            ),
            components=[
                Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))
            ],
        )

        block_Y = TensorBlock(
            values=Yval,
            samples=Labels(
                ["samples"], np.array([[0], [1], [2], [3], [4]], dtype=np.int32)
            ),
            components=[],
            properties=Labels(["properties"], np.array([[2]], dtype=np.int32)),
        )
        block_Y.add_gradient(
            "positions",
            data=Ygradval,
            samples=Labels(
                ["sample", "positions"],
                np.array([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]], dtype=np.int32),
            ),
            components=[
                Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))
            ],
        )

        keys = Labels(
            names=["key_1", "key_2"], values=np.array([[0, 0]], dtype=np.int32)
        )

        X = TensorMap(keys, [block_X])
        Y = TensorMap(keys, [block_Y])
        w = fn.lstsq(X, Y)

        self.assertTrue(len(w) == 1)
        self.assertTrue(np.all(w.keys == X.keys))
        self.assertTrue(
            np.allclose(w.block(0).values, np.array([1.0, 3.0]), rtol=1e-13)
        )

        for key, blockw in w:
            self.assertTrue(np.all(blockw.samples == Y.block(key).properties))
            self.assertTrue(np.all(blockw.properties == X.block(key).properties))

        Ydot = fn.dot(X, w)
        self.assertTrue(np.all(Ydot.keys == Y.keys))
        for key, expected_block in Ydot:
            comparing_dict = compare_blocks(expected_block, Y.block(key), rtol=1e-3)
            if not comparing_dict["general"]:
                print(str(comparing_dict))
            self.assertTrue(comparing_dict["general"])
            self.assertTrue(
                np.allclose(expected_block.values, Y.block(key).values, rtol=1e-13)
            )
            self.assertTrue(
                np.allclose(
                    expected_block.gradient("positions").data,
                    Y.block(key).gradient("positions").data,
                    rtol=1e-13,
                )
            )
        return


class TestDot(unittest.TestCase):
    def test_self_dot_no_components(self):
        tensor1 = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            use_numpy=True,
        )
        tensor2 = fn.remove_gradients(tensor1)
        dot_blocks = []
        for key, block1 in tensor1:
            block2 = tensor2.block(key)
            result_block = fn._dot_block(block1, block2)
            dot_blocks.append(result_block)
            expected_values = np.dot(block1.values, block2.values.T)
            self.assertTrue(
                np.allclose(result_block.values, expected_values, rtol=1e-13)
            )
            self.assertTrue(np.all(block1.samples == result_block.samples))
            self.assertTrue(np.all(block2.samples == result_block.properties))

            self.assertTrue(
                len(block1.gradients_list()) == len(result_block.gradients_list())
            )
            for parameter, gradient1 in block1.gradients():
                result_gradient = result_block.gradient(parameter)
                self.assertTrue(np.all(gradient1.samples == result_gradient.samples))
                self.assertTrue(
                    len(gradient1.components) == len(result_gradient.components)
                )
                for c1, cres in zip(gradient1.components, result_gradient.components):
                    self.assertTrue(np.all(c1 == cres))

                self.assertTrue(len(block2.samples) == len(result_gradient.properties))
                for p1, pres in zip(block2.samples, result_gradient.properties):
                    self.assertTrue(np.all(p1 == pres))

                expected_data = gradient1.data @ block2.values.T
                self.assertTrue(
                    np.allclose(expected_data, result_gradient.data, rtol=1e-13)
                )
        expected_tensor = TensorMap(tensor1.keys, dot_blocks)

        dot_tensor = fn.dot(tensor1=tensor1, tensor2=tensor2)
        self.assertTrue(np.all(expected_tensor.keys == dot_tensor.keys))
        for key, expected_block in expected_tensor:

            comparing_dict = compare_blocks(expected_block, dot_tensor.block(key))
            self.assertTrue(comparing_dict["general"])


class TestNormalize(unittest.TestCase):
    def test_normalize_no_components(self):
        data = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            use_numpy=True,
        )

        with self.assertRaises(ValueError) as cm:
            _ = fn.normalize_by_sample(data)

        self.assertEqual(
            str(cm.exception),
            "normalization of gradients w.r.t. 'cell' is not yet implemented",
        )

        data = fn.remove_gradients(data, remove=["cell"])
        normalized = fn.normalize_by_sample(data)

        for _, block in normalized:
            norm = np.linalg.norm(block.values, axis=-1)
            self.assertTrue(np.allclose(norm, np.ones_like(norm), rtol=1e-16))

        # TODO: add tests for gradients with finite differences

    def test_normalize_components(self):
        data = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            use_numpy=True,
        )

        with self.assertRaises(ValueError) as cm:
            _ = fn.normalize_by_sample(data)

        self.assertEqual(
            str(cm.exception),
            "normalization of equivariant tensors is not yet implemented",
        )


# TODO: add tests with torch & torch scripting/tracing

if __name__ == "__main__":
    unittest.main()
