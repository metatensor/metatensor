import os

import numpy as np

import equistore
import equistore.io
from equistore import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
TEST_FILE = "qm7-spherical-expansion.npz"


class TestAbs:
    """
    Unit tests for the :py:func:`equistore.abs` function.
    """

    def test_self_abs_tensor_nogradient(self):
        block_1 = TensorBlock(
            values=np.array([[1, -2], [-3, -5]]),
            samples=Labels(["samples"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )
        block_2 = TensorBlock(
            values=np.array([[-1, -2], [-3, 4], [5, -6]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )

        block_res1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["samples"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )
        block_res2 = TensorBlock(
            values=np.array([[1, 2], [3, 4], [5, 6]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )
        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
        A = TensorMap(keys, [block_1, block_2])
        A_copy = A.copy()
        tensor_abs = equistore.abs(A)
        tensor_result = TensorMap(keys, [block_res1, block_res2])

        assert equistore.equal(tensor_result, tensor_abs)
        assert equistore.equal(A, A_copy)

    def test_self_abs_tensor_gradient(self):
        block_1 = TensorBlock(
            values=np.array([[1, -2], [-3, -5]]),
            samples=Labels(["samples"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )
        block_1.add_gradient(
            "parameter",
            data=np.array([[[-6, -1], [7, -2]], [[-8, 3], [-9, -4]]]),
            samples=Labels(["sample", "positions"], np.array([[0, 1], [1, 1]])),
            components=[
                Labels.arange("components", 2),
            ],
        )

        block_2 = TensorBlock(
            values=np.array([[-1, -2], [-3, 4], [5, -6]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("properties", 2),
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
                np.array([[0, 1], [1, 1], [2, 1]]),
            ),
            components=[
                Labels.arange("components", 2),
            ],
        )

        block_res1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["samples"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )
        block_res1.add_gradient(
            "parameter",
            data=np.array([[[-6, 1], [7, 2]], [[8, -3], [9, 4]]]),
            samples=Labels(["sample", "positions"], np.array([[0, 1], [1, 1]])),
            components=[
                Labels.arange("components", 2),
            ],
        )
        block_res2 = TensorBlock(
            values=np.array([[1, 2], [3, 4], [5, 6]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("properties", 2),
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
                np.array([[0, 1], [1, 1], [2, 1]]),
            ),
            components=[
                Labels.arange("components", 2),
            ],
        )
        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
        A = TensorMap(keys, [block_1, block_2])
        tensor_abs = equistore.abs(A)
        tensor_result = TensorMap(keys, [block_res1, block_res2])

        assert equistore.equal(tensor_result, tensor_abs)

    def test_self_abs_tensor_components(self):
        A = equistore.load(os.path.join(DATA_ROOT, TEST_FILE), use_numpy=True)

        tensor_abs = equistore.abs(A)

        abs_blocks = []
        for _, block in A:
            new_block = TensorBlock(
                samples=block.samples,
                components=block.components,
                properties=block.properties,
                values=np.abs(block.values),
            )
            sign_values = np.sign(block.values)
            shape = ()
            for c in block.components:
                shape += (len(c),)
            shape += (len(block.properties),)
            for parameter, gradient in block.gradients():
                diff_components = len(gradient.components) - len(block.components)
                new_grad = gradient.data[:] * sign_values[
                    gradient.samples["sample"]
                ].reshape((-1,) + (1,) * diff_components + shape)
                new_block.add_gradient(
                    parameter,
                    new_grad,
                    gradient.samples,
                    gradient.components,
                )

            abs_blocks.append(new_block)

        tensor_result = TensorMap(A.keys, abs_blocks)
        assert equistore.equal(tensor_result, tensor_abs)

    def test_self_abs_tensor_complex(self):
        block_1 = TensorBlock(
            values=np.array([[1 + 2j, -2j], [5 - 3j, -5 + 5j]]),
            samples=Labels(["samples"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )
        block_2 = TensorBlock(
            values=np.array([[-1 + 1j, -2 + 6j], [4 - 3j, -4], [5j, 3]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )

        block_res1 = TensorBlock(
            values=np.array(
                [
                    [np.sqrt(1 + 2**2), 2],
                    [np.sqrt(5**2 + 3**2), np.sqrt(5**2 + 5**2)],
                ]
            ),
            samples=Labels(["samples"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )
        block_res2 = TensorBlock(
            values=np.array(
                [[np.sqrt(1 + 1), np.sqrt(2**2 + 6**2)], [5, 4], [5, 3]]
            ),
            samples=Labels(["samples"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )
        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
        A = TensorMap(keys, [block_1, block_2])
        tensor_abs = equistore.abs(A)
        tensor_result = TensorMap(keys, [block_res1, block_res2])
        assert equistore.allclose(tensor_result, tensor_abs)
