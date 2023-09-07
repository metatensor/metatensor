import os

import numpy as np
import pytest

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_self_multiply_tensors_nogradient():
    block_1 = TensorBlock(
        values=np.array([[1, 2], [3, 5]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_2 = TensorBlock(
        values=np.array([[1, 2], [3, 4], [5, 6]]),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_3 = TensorBlock(
        values=np.array([[1.5, 2.1], [6.7, 10.2]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_4 = TensorBlock(
        values=np.array([[10, 200.8], [3.76, 4.432], [545, 26]]),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels.range("p", 2),
    )

    block_res_1 = TensorBlock(
        values=np.array([[1.5, 4.2], [20.1, 51.0]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_res_2 = TensorBlock(
        values=np.array([[10.0, 401.6], [11.28, 17.728], [2725.0, 156.0]]),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
    A = TensorMap(keys, [block_1, block_2])
    B = TensorMap(keys, [block_3, block_4])
    A_copy = A.copy()
    B_copy = B.copy()
    tensor_sum = metatensor.multiply(A, B)
    tensor_result = TensorMap(keys, [block_res_1, block_res_2])

    assert metatensor.allclose(tensor_result, tensor_sum)
    # Check not modified in place
    assert metatensor.allclose(A, A_copy)
    assert metatensor.allclose(B, B_copy)


def test_self_multiply_tensors_gradient():
    block_1 = TensorBlock(
        values=np.array([[14, 24], [43, 45]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array([[[6, 1], [7, 2]], [[8, 3], [9, 4]]]),
            samples=Labels.range("sample", 2),
            components=[Labels.range("c", 2)],
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=np.array([[15, 25], [53, 54], [55, 65]]),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [[[10, 11], [12, 13]], [[14, 15], [10, 11]], [[12, 13], [14, 15]]]
            ),
            samples=Labels.range("sample", 3),
            components=[Labels.range("c", 2)],
            properties=block_2.properties,
        ),
    )

    block_3 = TensorBlock(
        values=np.array([[1.45, 2.41], [6.47, 10.42]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_3.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array([[[1, 0.1], [2, 0.2]], [[3, 0.3], [4.5, 0.4]]]),
            samples=Labels.range("sample", 2),
            components=[Labels.range("c", 2)],
            properties=block_3.properties,
        ),
    )

    block_4 = TensorBlock(
        values=np.array([[105, 200.58], [3.756, 4.4325], [545.5, 26.05]]),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_4.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[1.0, 1.1], [1.2, 1.3]],
                    [[1.4, 1.5], [1.0, 1.1]],
                    [[1.2, 1.3], [1.4, 1.5]],
                ]
            ),
            samples=Labels.range("sample", 3),
            components=[Labels.range("c", 2)],
            properties=block_4.properties,
        ),
    )

    block_res_1 = TensorBlock(
        values=np.array([[20.3, 57.84], [278.21, 468.9]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_res_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [[[22.7, 4.81], [38.15, 9.62]], [[180.76, 44.76], [251.73, 59.68]]]
            ),
            samples=Labels.range("sample", 2),
            components=[Labels.range("c", 2)],
            properties=block_res_1.properties,
        ),
    )

    block_res_2 = TensorBlock(
        values=np.array([[1575.0, 5014.5], [199.068, 239.355], [30002.5, 1693.25]]),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_res_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[1065.0, 2233.88], [1278.0, 2640.04]],
                    [[126.784, 147.4875], [90.56, 108.1575]],
                    [[6612.0, 423.15], [7714.0, 488.25]],
                ]
            ),
            samples=Labels.range("sample", 3),
            components=[Labels.range("c", 2)],
            properties=block_res_2.properties,
        ),
    )

    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
    A = TensorMap(keys, [block_1, block_2])
    B = TensorMap(keys, [block_3, block_4])
    A_copy = A.copy()
    B_copy = B.copy()
    tensor_sum = metatensor.multiply(A, B)
    tensor_result = TensorMap(keys, [block_res_1, block_res_2])

    assert metatensor.allclose(tensor_result, tensor_sum)
    # Check not modified in place
    assert metatensor.equal(A, A_copy)
    assert metatensor.equal(B, B_copy)


def test_self_multiply_scalar_gradient():
    block_1 = TensorBlock(
        values=np.array([[1, 2], [3, 5]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array([[[6, 1], [7, 2]], [[8, 3], [9, 4]]]),
            samples=Labels.range("sample", 2),
            components=[Labels.range("c", 2)],
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=np.array([[11, 12], [13, 14], [15, 16]]),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [[[10, 11], [12, 13]], [[14, 15], [10, 11]], [[12, 13], [14, 15]]]
            ),
            samples=Labels.range("sample", 3),
            components=[Labels.range("c", 2)],
            properties=block_2.properties,
        ),
    )

    block_res_1 = TensorBlock(
        values=np.array([[5.1, 10.2], [15.3, 25.5]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_res_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [[[30.6, 5.1], [35.7, 10.2]], [[40.8, 15.3], [45.9, 20.4]]]
            ),
            samples=Labels.range("sample", 2),
            components=[Labels.range("c", 2)],
            properties=block_res_1.properties,
        ),
    )

    block_res_2 = TensorBlock(
        values=np.array([[56.1, 61.2], [66.3, 71.4], [76.5, 81.6]]),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_res_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[51.0, 56.1], [61.2, 66.3]],
                    [[71.4, 76.5], [51.0, 56.1]],
                    [[61.2, 66.3], [71.4, 76.5]],
                ]
            ),
            samples=Labels.range("sample", 3),
            components=[Labels.range("c", 2)],
            properties=block_res_2.properties,
        ),
    )

    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
    A = TensorMap(keys, [block_1, block_2])
    B = 5.1

    tensor_sum = metatensor.multiply(A, B)
    tensor_result = TensorMap(keys, [block_res_1, block_res_2])

    assert metatensor.allclose(tensor_result, tensor_sum)

    def test_self_multiply_tensors_gradient_additional_components(self):
        block_1 = TensorBlock(
            values=np.array([[1]]),
            samples=Labels(["s"], np.array([[2]])),
            components=[],
            properties=Labels.range("p", 1),
        )
        block_1_grad = TensorBlock(
            values=np.array([[[1], [3]], [[5], [7]]]),
            samples=Labels(["sample", "g"], np.array([[0, 0], [0, 1]])),
            components=[Labels.range("c", 2)],
            properties=Labels.range("p", 1),
        )
        block_2 = TensorBlock(
            values=np.array([[2]]),
            samples=Labels(["s"], np.array([[2]])),
            components=[],
            properties=Labels.range("p", 1),
        )
        block_2_grad = TensorBlock(
            values=np.array([[[0], [2]], [[4], [6]]]),
            samples=Labels(["sample", "g"], np.array([[0, 0], [0, 1]])),
            components=[Labels.range("c", 2)],
            properties=Labels.range("p", 1),
        )
        block_1.add_gradient("g", block_1_grad)
        block_2.add_gradient("g", block_2_grad)
        keys = Labels(names=["key"], values=np.array([[0]]))
        tensor_1 = TensorMap(keys, [block_1])
        tensor_2 = TensorMap(keys, [block_2])

        block_result = TensorBlock(
            values=np.array([[2]]),
            samples=Labels(["s"], np.array([[2]])),
            components=[],
            properties=Labels.range("p", 1),
        )
        block_result_grad = TensorBlock(
            values=np.array([[[2], [8]], [[14], [20]]]),
            samples=Labels(["sample", "g"], np.array([[0, 0], [0, 1]])),
            components=[Labels.range("c", 2)],
            properties=Labels.range("p", 1),
        )
        block_result.add_gradient("g", block_result_grad)
        tensor_result = TensorMap(keys, [block_result])
        product_tensor = metatensor.multiply(tensor_1, tensor_2)
        assert metatensor.equal(tensor_result, product_tensor)


def test_self_multiply_error():
    block = TensorBlock(
        values=np.array([[1, 2], [3, 5]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0]]))
    tensor = TensorMap(keys, [block])

    message = "`A` must be a metatensor TensorMap, not <class 'numpy.ndarray'>"
    with pytest.raises(TypeError, match=message):
        keys = metatensor.multiply(np.ones((3, 4)), tensor)

    message = (
        "`B` must be a metatensor TensorMap or a scalar value, "
        "not <class 'numpy.ndarray'>"
    )
    with pytest.raises(TypeError, match=message):
        keys = metatensor.multiply(tensor, np.ones((3, 4)))
