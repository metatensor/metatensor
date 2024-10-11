import os

import numpy as np

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_self_dot_no_components():
    tensor_1 = metatensor.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"))
    tensor_2 = metatensor.remove_gradients(tensor_1)

    dot_tensor = metatensor.dot(tensor_1=tensor_1, tensor_2=tensor_2)
    assert tensor_1.keys == dot_tensor.keys

    for key, block_1 in tensor_1.items():
        dot_block = dot_tensor.block(key)

        block_2 = tensor_2.block(key)
        expected_values = np.dot(block_1.values, block_2.values.T)

        assert np.allclose(dot_block.values, expected_values, rtol=1e-13)
        assert block_1.samples == dot_block.samples
        assert block_2.samples == dot_block.properties

        assert block_1.gradients_list() == dot_block.gradients_list()

        for parameter, gradient_1 in block_1.gradients():
            result_gradient = dot_block.gradient(parameter)
            assert gradient_1.samples == result_gradient.samples

            assert gradient_1.components == result_gradient.components
            assert block_2.samples == result_gradient.properties

            expected_gradient_values = gradient_1.values @ block_2.values.T

            assert np.allclose(
                expected_gradient_values, result_gradient.values, rtol=1e-13
            )


def test_self_dot_components():
    tensor_1 = metatensor.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"))

    tensor_2 = []
    n_samples = 42
    for block_1 in tensor_1:
        value2 = np.arange(n_samples * len(block_1.properties)).reshape(
            n_samples, len(block_1.properties)
        )

        block_2 = TensorBlock(
            values=value2,
            samples=Labels(
                ["s"],
                np.array([[i] for i in range(n_samples)]),
            ),
            components=[],
            properties=block_1.properties,
        )
        tensor_2.append(block_2)

    tensor_2 = TensorMap(tensor_1.keys, tensor_2)

    dot_tensor = metatensor.dot(tensor_1=tensor_1, tensor_2=tensor_2)
    assert tensor_1.keys == dot_tensor.keys

    for key, block_1 in tensor_1.items():
        block_2 = tensor_2.block(key)
        dot_block = dot_tensor.block(key)

        expected_values = np.dot(
            block_1.values,
            block_2.values.T,
        )
        assert np.allclose(dot_block.values, expected_values, rtol=1e-13)

        assert block_1.samples == dot_block.samples
        assert block_2.samples == dot_block.properties

        assert block_1.gradients_list() == dot_block.gradients_list()

        for parameter, gradient_1 in block_1.gradients():
            result_gradient = dot_block.gradient(parameter)
            assert gradient_1.samples == result_gradient.samples

            assert gradient_1.components == result_gradient.components

            assert block_2.samples == result_gradient.properties

            expected_gradient_values = gradient_1.values @ value2.T
            assert np.allclose(
                expected_gradient_values, result_gradient.values, rtol=1e-13
            )
