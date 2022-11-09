import os
import unittest
import numpy as np
from utils import compare_blocks

import equistore.operations as fn
from equistore import Labels, TensorBlock, TensorMap

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


class TestLstsq(unittest.TestCase):
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
        w = fn.lstsq(X, Y, rcond=1e-13)

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
        w = fn.lstsq(X, Y, rcond=1e-13)

        self.assertTrue(len(w) == 1)
        self.assertTrue(np.all(w.keys == X.keys))
        self.assertTrue(
            np.allclose(w.block(0).values, np.array([1.0, 3.0]), rtol=1e-13)
        )

        for key, block_w in w:
            self.assertTrue(np.all(block_w.samples == Y.block(key).properties))
            self.assertTrue(np.all(block_w.properties == X.block(key).properties))

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

    def test_self_lstsq_grad_components(self):
        Xval, Xgradval, Yval, Ygradval = get_value_linear_solve()
        xdim = len(Xval)
        ydim = len(Yval)
        block_X = TensorBlock(
            values=Xval.reshape((1, xdim, Xval.shape[-1])),
            samples=Labels(["samples"], np.array([[0]], dtype=np.int32)),
            components=[
                Labels(
                    ["components"], np.array([[0], [1], [2], [3], [4]]), dtype=np.int32
                )
            ],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_X.add_gradient(
            "positions",
            data=Xgradval.reshape(1, 3, xdim, Xval.shape[-1]),
            samples=Labels(
                ["sample", "positions"],
                np.array([[0, 1]], dtype=np.int32),
            ),
            components=[
                Labels(["der_components"], np.array([[0], [1], [2]], dtype=np.int32)),
                Labels(
                    ["components"], np.array([[0], [1], [2], [3], [4]], dtype=np.int32)
                ),
            ],
        )

        block_Y = TensorBlock(
            values=Yval.reshape((1, ydim, Yval.shape[-1])),
            samples=Labels(["samples"], np.array([[0]], dtype=np.int32)),
            components=[
                Labels(
                    ["components"], np.array([[0], [1], [2], [3], [4]]), dtype=np.int32
                )
            ],
            properties=Labels(["properties"], np.array([[2]], dtype=np.int32)),
        )
        block_Y.add_gradient(
            "positions",
            data=Ygradval.reshape((1, 3, ydim, Yval.shape[-1])),
            samples=Labels(
                ["sample", "positions"],
                np.array([[0, 1]], dtype=np.int32),
            ),
            components=[
                Labels(["der_components"], np.array([[0], [1], [2]], dtype=np.int32)),
                Labels(
                    ["components"], np.array([[0], [1], [2], [3], [4]], dtype=np.int32)
                ),
            ],
        )

        keys = Labels(
            names=["key_1", "key_2"], values=np.array([[0, 0]], dtype=np.int32)
        )

        X = TensorMap(keys, [block_X])
        Y = TensorMap(keys, [block_Y])
        w = fn.lstsq(X, Y, rcond=1e-13)

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


def Xfun1(x, y, z):
    return np.arctan(-x + 2 * y * y + 3 * z * z * z)


def Xfun1_dx(x, y, z):
    """derivative w.r.t x of Xfun1"""
    return -1 / (1 + (-x + 2 * y * y + 3 * z * z * z) ** 2)


def Xfun1_dy(x, y, z):
    """derivative w.r.t y of Xfun1"""
    return 4 * y / (1 + (-x + 2 * y * y + 3 * z * z * z) ** 2)


def Xfun1_dz(x, y, z):
    """derivative w.r.t z of Xfun1"""
    return 9 * z * z / (1 + (-x + 2 * y * y + 3 * z * z * z) ** 2)


def Xfun2(x, y, z):
    return x**3 + 2 * y + 3 * z**2


def Xfun2_dx(x, y, z):
    """derivative w.r.t x of Xfun2"""
    return 3 * x**2


def Xfun2_dy(x, y, z):
    """derivative w.r.t y of Xfun2"""
    return 2


def Xfun2_dz(x, y, z):
    """derivative w.r.t z of Xfun2"""
    return 6 * z


def get_value_linear_solve():
    """Generate a value matrix for block and gradient in
    the test for the linear solve
    """
    data = np.arange(15).reshape((-1, 3))
    Xval = np.zeros((len(data), 2))
    Xgradval = np.zeros((len(data), 3, 2))
    for i in range(len(data)):
        Xval[i, 0] = Xfun1(data[i, 0], data[i, 1], data[i, 2])
        Xval[i, 1] = Xfun2(data[i, 0], data[i, 1], data[i, 2])
        Xgradval[i, 0, 0] = Xfun1_dx(data[i, 0], data[i, 1], data[i, 2])
        Xgradval[i, 1, 0] = Xfun1_dy(data[i, 0], data[i, 1], data[i, 2])
        Xgradval[i, 2, 0] = Xfun1_dz(data[i, 0], data[i, 1], data[i, 2])
        Xgradval[i, 0, 1] = Xfun2_dx(data[i, 0], data[i, 1], data[i, 2])
        Xgradval[i, 1, 1] = Xfun2_dy(data[i, 0], data[i, 1], data[i, 2])
        Xgradval[i, 2, 1] = Xfun2_dz(data[i, 0], data[i, 1], data[i, 2])

    w = np.array([[1], [3]])
    Yval = np.dot(Xval, w)
    Ygradval = np.dot(Xgradval, w)

    return Xval, Xgradval, Yval, Ygradval


# TODO: add tests with torch & torch scripting/tracing

if __name__ == "__main__":
    unittest.main()
