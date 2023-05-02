import numpy as np
import pytest

import equistore
from equistore import Labels, TensorMap


# import torch
# from numpy.testing import assert_equal


@pytest.fixture
def block():
    # Returns a block with nested gradients
    values = np.arange(42, dtype=np.float64).reshape(7, 3, 2)
    grad_values = np.arange(42, dtype=np.float64).reshape(7, 3, 2)
    grad_grad_values = np.arange(42, dtype=np.float64).reshape(7, 3, 2)
    block = equistore.block_from_array(values)
    grad_block = equistore.block_from_array(grad_values)
    grad_grad_block = equistore.block_from_array(grad_grad_values)
    grad_block.add_gradient("grad_grad", grad_grad_block)
    block.add_gradient("grad", grad_block)
    return block


@pytest.fixture
def tmap():
    # Returns a TensorMap with two blocks
    values1 = np.arange(42, dtype=np.float64).reshape(7, 3, 2)
    block1 = equistore.block_from_array(values1)
    values2 = np.arange(100, dtype=np.float64).reshape(10, 5, 2)
    block2 = equistore.block_from_array(values2)
    tmap = TensorMap(keys=Labels.arange(2), blocks=[block1, block2])
    return tmap


def test_wrong_arguments_block(block):
    """Test the `to` function with incorrect arguments."""
    with pytest.raises(
        TypeError, match="``block`` should be an equistore `TensorBlock`"
    ):
        equistore.block_to(100, backend="numpy")
    with pytest.raises(TypeError, match="`backend` should be passed as a `str`"):
        equistore.block_to(block, backend=10)
    with pytest.raises(
        ValueError, match="The `numpy` backend option does not support gradients"
    ):
        equistore.block_to(block, backend="numpy", requires_grad=True)
    with pytest.raises(ValueError, match="not supported"):
        equistore.block_to(block, backend="jax")


def test_numpy_to_torch_block(block):
    """Test a conversion from numpy to torch."""
    _ = equistore.block_to(block, backend="torch")
    # assert equistore.equal_metadata_block(block, new_block)


def test_torch_to_numpy_block(block):
    """Test error with unknown `axis` keyword."""


def test_torch_to_gpu_block(block):
    """Test error with unknown `axis` keyword."""


def test_change_dtype_block(block):
    """Test error with unknown `axis` keyword."""


def test_wrong_arguments(block):
    """Test error with unknown `axis` keyword."""


def test_numpy_to_torch(block):
    """Test error with unknown `axis` keyword."""


def test_torch_to_numpy(block):
    """Test error with unknown `axis` keyword."""


def test_torch_to_gpu(block):
    """Test error with unknown `axis` keyword."""


def test_change_dtype(block):
    """Test error with unknown `axis` keyword."""
