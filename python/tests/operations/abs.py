import os

import numpy as np

import equistore
import equistore.io
from equistore import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
TEST_FILE = "qm7-spherical-expansion.npz"


class TestAbs:
    def test_self_abs_tensor_nogradient(self):
        block_1 = TensorBlock(
            values=np.array([[1, -2], [-3, -5]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_2 = TensorBlock(
            values=np.array([[-1, -2], [-3, 4], [5, -6]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )

        block_res1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_res2 = TensorBlock(
            values=np.array([[1, 2], [3, 4], [5, 6]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        keys = Labels(
            names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]], dtype=np.int32)
        )
        A = TensorMap(keys, [block_1, block_2])
        tensor_abs = equistore.abs(A)
        tensor_result = TensorMap(keys, [block_res1, block_res2])

        assert equistore.equal(tensor_result, tensor_abs)

    def test_self_abs_tensor_gradient(self):
        block_1 = TensorBlock(
            values=np.array([[1, -2], [-3, -5]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_1.add_gradient(
            "parameter",
            data=np.array([[[-6, -1], [7, -2]], [[-8, 3], [-9, -4]]]),
            samples=Labels(
                ["sample", "positions"], np.array([[0, 1], [1, 1]], dtype=np.int32)
            ),
            components=[
                Labels(["components"], np.array([[0], [1]], dtype=np.int32)),
            ],
        )

        block_2 = TensorBlock(
            values=np.array([[-1, -2], [-3, 4], [5, -6]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_2.add_gradient(
            "parameter",
            data=np.array(
                [
                    [[-10, 11], [-12, 13]],
                    [[-14, -15], [10, -11]],
                    [[-12, -13], [14, -15]],
                ]
            ),
            samples=Labels(
                ["sample", "positions"],
                np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int32),
            ),
            components=[
                Labels(["components"], np.array([[0], [1]], dtype=np.int32)),
            ],
        )

        block_res1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_res1.add_gradient(
            "parameter",
            data=np.array([[[-6, 1], [7, 2]], [[8, -3], [9, 4]]]),
            samples=Labels(
                ["sample", "positions"], np.array([[0, 1], [1, 1]], dtype=np.int32)
            ),
            components=[
                Labels(["components"], np.array([[0], [1]], dtype=np.int32)),
            ],
        )
        block_res2 = TensorBlock(
            values=np.array([[1, 2], [3, 4], [5, 6]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_res2.add_gradient(
            "parameter",
            data=np.array(
                [
                    [[10, -11], [12, -13]],
                    [[14, -15], [-10, -11]],
                    [[-12, 13], [14, 15]],
                ]
            ),
            samples=Labels(
                ["sample", "positions"],
                np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int32),
            ),
            components=[
                Labels(["components"], np.array([[0], [1]], dtype=np.int32)),
            ],
        )
        keys = Labels(
            names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]], dtype=np.int32)
        )
        A = TensorMap(keys, [block_1, block_2])
        tensor_abs = equistore.abs(A)
        tensor_result = TensorMap(keys, [block_res1, block_res2])

        equistore.equal_raise(tensor_result, tensor_abs)
        assert equistore.equal(tensor_result, tensor_abs)
