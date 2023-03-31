import os

import numpy as np
import pytest

import equistore
from equistore import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


class TestAdd:
    @pytest.fixture
    def keys(self):
        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
        return keys

    @pytest.fixture
    def tensor_A(self, keys):
        block_1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["samples"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )

        block_1.add_gradient(
            "parameter",
            data=np.array([[[6, 1], [7, 2]], [[8, 3], [9, 4]]]),
            samples=Labels(["sample", "positions"], np.array([[0, 1], [1, 1]])),
            components=[
                Labels.arange("components", 2),
            ],
        )

        block_2 = TensorBlock(
            values=np.array([[1, 2], [3, 4], [5, 6]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )

        block_2.add_gradient(
            "parameter",
            data=np.array(
                [[[10, 11], [12, 13]], [[14, 15], [10, 11]], [[12, 13], [14, 15]]]
            ),
            samples=Labels(
                ["sample", "positions"],
                np.array([[0, 1], [1, 1], [2, 1]]),
            ),
            components=[
                Labels.arange("components", 2),
            ],
        )

        return TensorMap(keys, [block_1, block_2])

    @pytest.fixture
    def tensor_B(self, keys):
        block_1 = TensorBlock(
            values=np.array([[1.5, 2.1], [6.7, 10.2]]),
            samples=Labels(["samples"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )

        block_1.add_gradient(
            "parameter",
            data=np.array([[[1, 0.1], [2, 0.2]], [[3, 0.3], [4.5, 0.4]]]),
            samples=Labels(["sample", "positions"], np.array([[0, 1], [1, 1]])),
            components=[
                Labels.arange("components", 2),
            ],
        )

        block_2 = TensorBlock(
            values=np.array([[10, 200.8], [3.76, 4.432], [545, 26]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )

        block_2.add_gradient(
            "parameter",
            data=np.array(
                [
                    [[1.0, 1.1], [1.2, 1.3]],
                    [[1.4, 1.5], [1.0, 1.1]],
                    [[1.2, 1.3], [1.4, 1.5]],
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

        return TensorMap(keys, [block_1, block_2])

    @pytest.fixture
    def tensor_res_1(self, keys):
        block_1 = TensorBlock(
            values=np.array([[2.5, 4.1], [9.7, 15.2]]),
            samples=Labels(["samples"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )

        block_1.add_gradient(
            "parameter",
            data=np.array(np.array([[[7, 1.1], [9, 2.2]], [[11, 3.3], [13.5, 4.4]]])),
            samples=Labels(["sample", "positions"], np.array([[0, 1], [1, 1]])),
            components=[
                Labels.arange("components", 2),
            ],
        )

        block_2 = TensorBlock(
            values=np.array([[11, 202.8], [6.76, 8.432], [550, 32]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )

        block_2.add_gradient(
            "parameter",
            data=np.array(
                [
                    [[11.0, 12.1], [13.2, 14.3]],
                    [[15.4, 16.5], [11.0, 12.1]],
                    [[13.2, 14.3], [15.4, 16.5]],
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

        return TensorMap(keys, [block_1, block_2])

    @pytest.fixture
    def tensor_res_2(self, keys):
        block_1 = TensorBlock(
            values=np.array([[6.1, 7.1], [8.1, 10.1]]),
            samples=Labels(["samples"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )
        block_1.add_gradient(
            "parameter",
            data=np.array([[[6, 1], [7, 2]], [[8, 3], [9, 4]]]),
            samples=Labels(["sample", "positions"], np.array([[0, 1], [1, 1]])),
            components=[
                Labels.arange("components", 2),
            ],
        )
        block_2 = TensorBlock(
            values=np.array([[6.1, 7.1], [8.1, 9.1], [10.1, 11.1]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("properties", 2),
        )
        block_2.add_gradient(
            "parameter",
            data=np.array(
                [[[10, 11], [12, 13]], [[14, 15], [10, 11]], [[12, 13], [14, 15]]]
            ),
            samples=Labels(
                ["sample", "positions"],
                np.array([[0, 1], [1, 1], [2, 1]]),
            ),
            components=[
                Labels.arange("components", 2),
            ],
        )

        return TensorMap(keys, [block_1, block_2])

    def test_self_add_tensors_nogradient(self, tensor_A, tensor_B, tensor_res_1):
        tensor_A = equistore.remove_gradients(tensor_A)
        tensor_B = equistore.remove_gradients(tensor_B)
        tensor_res_1 = equistore.remove_gradients(tensor_res_1)

        tensor_A_copy = tensor_A.copy()
        tensor_B_copy = tensor_B.copy()

        equistore.allclose_raise(equistore.add(tensor_A, tensor_B), tensor_res_1)

        # Check the tensors haven't be modified in place
        equistore.equal_raise(tensor_A, tensor_A_copy)
        equistore.equal_raise(tensor_B, tensor_B_copy)

    def test_self_add_tensors_gradient(self, tensor_A, tensor_B, tensor_res_1):
        equistore.allclose_raise(equistore.add(tensor_A, tensor_B), tensor_res_1)

    @pytest.mark.parametrize("tensor_B", [5.1, np.array([5.1])])
    def test_self_add_scalar_gradient(self, tensor_A, tensor_B, tensor_res_2):
        tensor_A_copy = tensor_A.copy()

        equistore.allclose_raise(equistore.add(tensor_A, tensor_B), tensor_res_2)

        # Check the tensors haven't be modified in place
        equistore.equal_raise(tensor_A, tensor_A_copy)

    def test_self_add_error(self, tensor_A):
        msg = "B should be a TensorMap or a scalar value."
        with pytest.raises(TypeError, match=msg):
            equistore.add(tensor_A, np.ones((3, 4)))


# TODO: add tests with torch & torch scripting/tracing
