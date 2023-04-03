"""Unit tests for the :py:func:`equistore.abs` function."""

import numpy as np
import pytest

import equistore
from equistore import TensorBlock, TensorMap

from .. import utils


@pytest.fixture
def tensor():
    """A pytest fixture for the test function :func:`utils.tensor`."""
    return utils.tensor()


@pytest.fixture
def tensor_map_abs(tensor):
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
                data=-gradient.data - 2j,
                samples=gradient.samples,
                components=gradient.components,
            )

        blocks.append(new_block)

    return TensorMap(keys=tensor.keys, blocks=blocks)


@pytest.fixture
def tensor_map_result(tensor_map_abs):
    blocks = []
    for _, block in tensor_map_abs:
        new_block = TensorBlock(
            values=np.abs(block.values),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                data=-gradient.data,
                samples=gradient.samples,
                components=gradient.components,
            )

        blocks.append(new_block)

    return TensorMap(keys=tensor_map_abs.keys, blocks=blocks)


@pytest.mark.parametrize("gradients", (True, False))
def test_abs(gradients, tensor_map_abs, tensor_map_result):
    if not gradients:
        tensor_map_abs = equistore.remove_gradients(tensor_map_abs)
        tensor_map_result = equistore.remove_gradients(tensor_map_result)

    tensor_map_copy = tensor_map_abs.copy()

    equistore.equal_raise(equistore.abs(tensor_map_abs), tensor_map_result)

    # Check the tensors haven't be modified in place
    equistore.equal_raise(tensor_map_abs, tensor_map_copy)
