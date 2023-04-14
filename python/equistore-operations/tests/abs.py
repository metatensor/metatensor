"""Unit tests for the :py:func:`equistore.abs` function."""

import numpy as np
import pytest

import equistore
from equistore import TensorBlock, TensorMap

from . import utils


@pytest.fixture
def tensor():
    return utils.tensor()


@pytest.fixture
def tensor_map_complex(tensor):
    """Manipulate tensor to be a suitable test for the `abs` function."""
    blocks = []
    for _, block in tensor:
        new_block = TensorBlock(
            values=-block.values + 3j,
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=-gradient.values - 2j,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=gradient.properties,
                ),
            )

        blocks.append(new_block)

    return TensorMap(keys=tensor.keys, blocks=blocks)


@pytest.fixture
def tensor_map_result(tensor_map_complex):
    blocks = []
    for _, block in tensor_map_complex:
        new_block = TensorBlock(
            values=np.abs(block.values),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=-gradient.values,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=gradient.properties,
                ),
            )

        blocks.append(new_block)

    return TensorMap(keys=tensor_map_complex.keys, blocks=blocks)


@pytest.mark.parametrize("gradients", (True, False))
def test_abs(gradients, tensor_map_complex, tensor_map_result):
    if not gradients:
        tensor_map_complex = equistore.remove_gradients(tensor_map_complex)
        tensor_map_result = equistore.remove_gradients(tensor_map_result)

    tensor_map_copy = tensor_map_complex.copy()

    tensor_abs = equistore.abs(tensor_map_complex)
    assert equistore.equal(tensor_abs, tensor_map_result)

    # Check the tensors haven't be modified in place
    assert equistore.equal(tensor_map_complex, tensor_map_copy)
