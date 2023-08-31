import numpy as np
import pytest

import metatensor
from metatensor import Labels, TensorBlock

from . import utils


@pytest.fixture
def block():
    return TensorBlock(
        values=np.full((3, 2), -1.0),
        samples=Labels(["s"], np.array([[0], [2], [4]])),
        components=[],
        properties=Labels(["p"], np.array([[5], [3]])),
    )


@pytest.fixture
def block_components():
    return TensorBlock(
        values=np.full((3, 3, 2, 2), -1.0),
        samples=Labels(["s"], np.array([[0], [2], [4]])),
        components=[
            Labels(["c_1"], np.array([[-1], [0], [1]])),
            Labels(["c_2"], np.array([[-4], [1]])),
        ],
        properties=Labels(["p"], np.array([[5], [3]])),
    )


@pytest.fixture
def tensor():
    return utils.tensor()


@pytest.fixture
def large_tensor():
    return utils.large_tensor()


def test_block_eq(block):
    assert metatensor.equal_block(block, block) == (block == block)


def test_block_neq(block, block_components):
    assert metatensor.equal_block(block, block_components) == (
        block == block_components
    )


def test_tensor_eq(tensor):
    assert metatensor.equal(tensor, tensor) == (tensor == tensor)


def test_tensor_neq(tensor, large_tensor):
    assert metatensor.equal(tensor, large_tensor) == (tensor == large_tensor)


def test_tensor_add(tensor):
    assert metatensor.add(tensor, 1) == (tensor + 1)


def test_tensor_sub(tensor):
    assert metatensor.subtract(tensor, 1) == (tensor - 1)


def test_tensor_mul(tensor):
    assert metatensor.multiply(tensor, 2) == (tensor * 2)


def test_tensor_matmul(tensor):
    tensor = tensor.components_to_properties("c")
    tensor = metatensor.remove_gradients(tensor)

    assert metatensor.dot(tensor, tensor) == (tensor @ tensor)


def test_tensor_truediv(tensor):
    assert metatensor.divide(tensor, 2) == (tensor / 2)


def test_tensor_pow(tensor):
    assert metatensor.pow(tensor, 2) == (tensor**2)
