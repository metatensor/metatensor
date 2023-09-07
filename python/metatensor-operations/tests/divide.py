import numpy as np
import pytest

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


def test_self_divide_tensors_no_gradient():
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

    block_res1 = TensorBlock(
        values=block_1.values[:] / block_3.values[:],
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_res2 = TensorBlock(
        values=block_2.values[:] / block_4.values[:],
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
    A = TensorMap(keys, [block_1, block_2])
    A_copy = A.copy()
    B = TensorMap(keys, [block_3, block_4])
    B_copy = B.copy()
    tensor_sum = metatensor.divide(A, B)
    tensor_result = TensorMap(keys, [block_res1, block_res2])

    assert metatensor.allclose(tensor_result, tensor_sum)

    assert metatensor.equal(A, A_copy)
    assert metatensor.equal(B, B_copy)


def test_self_divide_tensors_gradient():
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
            samples=Labels(["sample", "g"], np.array([[0, 1], [1, 1]])),
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
            samples=Labels(["sample", "g"], np.array([[0, 1], [1, 1], [2, 1]])),
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
            samples=Labels(["sample", "g"], np.array([[0, 1], [1, 1]])),
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
            samples=Labels(["sample", "g"], np.array([[0, 1], [1, 1], [2, 1]])),
            components=[Labels.range("c", 2)],
            properties=block_4.properties,
        ),
    )

    block_res1 = TensorBlock(
        values=np.array([[9.65517241, 9.95850622], [6.64605873, 4.31861804]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_res1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[-2.52080856, 1.72173344e-03], [-8.48989298, 3.44346688e-03]],
                    [[-1.84515861, 0.16357146], [-3.23141643, 0.21809528]],
                ]
            ),
            samples=Labels(["sample", "g"], np.array([[0, 1], [1, 1]])),
            components=[Labels.range("c", 2)],
            properties=block_res1.properties,
        ),
    )
    block_res2 = TensorBlock(
        values=np.array(
            [
                [0.14285714, 0.12463855],
                [14.11075612, 12.18274112],
                [0.10082493, 2.49520154],
            ]
        ),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_res2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[0.09387755, 0.05415743], [0.11265306, 0.06400424]],
                    [[-1.53223072, -0.73866028], [-1.09445051, -0.5416842]],
                    [[0.02177637, 0.37451969], [0.02540577, 0.43213811]],
                ]
            ),
            samples=Labels(["sample", "g"], np.array([[0, 1], [1, 1], [2, 1]])),
            components=[Labels.range("c", 2)],
            properties=block_res2.properties,
        ),
    )

    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
    A = TensorMap(keys, [block_1, block_2])
    A_copy = A.copy()
    B = TensorMap(keys, [block_3, block_4])
    B_copy = B.copy()
    tensor_sum = metatensor.divide(A, B)
    tensor_result = TensorMap(keys, [block_res1, block_res2])

    assert metatensor.allclose(
        tensor_result,
        tensor_sum,
        atol=1e-8,
    )

    assert metatensor.equal(A, A_copy)
    assert metatensor.equal(B, B_copy)


def test_self_divide_scalar_gradient():
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
            samples=Labels(["sample", "g"], np.array([[0, 1], [1, 1]])),
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
            samples=Labels(["sample", "g"], np.array([[0, 1], [1, 1], [2, 1]])),
            components=[Labels.range("c", 2)],
            properties=block_2.properties,
        ),
    )

    block_res1 = TensorBlock(
        values=np.array([[0.19607843, 0.39215686], [0.58823529, 0.98039216]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_res1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[1.17647059, 0.19607843], [1.37254902, 0.39215686]],
                    [[1.56862745, 0.58823529], [1.76470588, 0.78431373]],
                ]
            ),
            samples=Labels(["sample", "g"], np.array([[0, 1], [1, 1]])),
            components=[Labels.range("c", 2)],
            properties=block_res1.properties,
        ),
    )
    block_res2 = TensorBlock(
        values=np.array(
            [
                [2.15686275, 2.35294118],
                [2.54901961, 2.74509804],
                [2.94117647, 3.1372549],
            ]
        ),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_res2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[1.96078431, 2.15686275], [2.35294118, 2.54901961]],
                    [[2.74509804, 2.94117647], [1.96078431, 2.15686275]],
                    [[2.35294118, 2.54901961], [2.74509804, 2.94117647]],
                ]
            ),
            samples=Labels(["sample", "g"], np.array([[0, 1], [1, 1], [2, 1]])),
            components=[Labels.range("c", 2)],
            properties=block_res2.properties,
        ),
    )

    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
    A = TensorMap(keys, [block_1, block_2])
    B = 5.1

    tensor_sum = metatensor.divide(A, B)
    tensor_result = TensorMap(keys, [block_res1, block_res2])

    assert metatensor.allclose(tensor_result, tensor_sum, rtol=1e-8)

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
            values=np.array([[0.5]]),
            samples=Labels(["s"], np.array([[2]])),
            components=[],
            properties=Labels.range("p", 1),
        )
        block_result_grad = TensorBlock(
            values=np.array([[[0.5], [1.0]], [[1.5], [2.0]]]),
            samples=Labels(["sample", "g"], np.array([[0, 0], [0, 1]])),
            components=[Labels.range("c", 2)],
            properties=Labels.range("p", 1),
        )
        block_result.add_gradient("g", block_result_grad)
        tensor_result = TensorMap(keys, [block_result])
        quotient_tensor = metatensor.divide(tensor_1, tensor_2)
        assert metatensor.equal(tensor_result, quotient_tensor)


def test_self_divide_error():
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
        metatensor.divide(np.ones((3, 4)), tensor)

    message = (
        "`B` must be a metatensor TensorMap or a scalar value, "
        "not <class 'numpy.ndarray'>"
    )
    with pytest.raises(TypeError, match=message):
        metatensor.divide(tensor, np.ones((3, 4)))
