import numpy as np
import pytest
import torch
from numpy.testing import assert_equal

import equistore
from equistore import Labels, TensorMap


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
def tensor():
    # Returns a TensorMap with two blocks
    values1 = np.arange(42, dtype=np.float64).reshape(7, 3, 2)
    block1 = equistore.block_from_array(values1)
    values2 = np.arange(100, dtype=np.float64).reshape(10, 5, 2)
    block2 = equistore.block_from_array(values2)
    tensor = TensorMap(keys=Labels.arange("dummy", 2), blocks=[block1, block2])
    return tensor


def test_wrong_arguments_block(block):
    """Test the `block_to` function with incorrect arguments."""
    with pytest.raises(TypeError, match="`block` should be an equistore `TensorBlock`"):
        equistore.block_to(100)
    with pytest.raises(TypeError, match="`backend` should be passed as a `str`"):
        equistore.block_to(block, backend=10)
    with pytest.raises(
        ValueError,
        match="the `numpy` backend option does not support autograd gradient tracking",
    ):
        equistore.block_to(block, backend="numpy", requires_grad=True)
    with pytest.raises(ValueError, match="not supported"):
        equistore.block_to(block, backend="jax")


def test_numpy_to_torch_block(block):
    """Test a `block_to` conversion from numpy to torch."""
    new_block = equistore.block_to(block, backend="torch")
    assert_equal(new_block.properties, block.properties)
    assert_equal(new_block.gradient("grad").samples, block.gradient("grad").samples)
    assert_equal(
        new_block.gradient("grad").gradient("grad_grad").components,
        block.gradient("grad").gradient("grad_grad").components,
    )
    assert isinstance(new_block.values, torch.Tensor)
    assert isinstance(new_block.gradient("grad").values, torch.Tensor)
    assert isinstance(
        new_block.gradient("grad").gradient("grad_grad").values, torch.Tensor
    )
    assert_equal(new_block.values.numpy(), block.values)
    assert_equal(
        new_block.gradient("grad").values.numpy(), block.gradient("grad").values
    )
    assert_equal(
        new_block.gradient("grad").gradient("grad_grad").values.numpy(),
        block.gradient("grad").gradient("grad_grad").values,
    )


def test_torch_to_numpy_block(block):
    """Test a `block_to` conversion from torch to numpy."""
    block = equistore.block_to(block, backend="torch")
    new_block = equistore.block_to(block, backend="numpy")
    assert_equal(new_block.samples, block.samples)
    assert_equal(
        new_block.gradient("grad").components, block.gradient("grad").components
    )
    assert_equal(
        new_block.gradient("grad").gradient("grad_grad").properties,
        block.gradient("grad").gradient("grad_grad").properties,
    )
    assert isinstance(new_block.values, np.ndarray)
    assert isinstance(new_block.gradient("grad").values, np.ndarray)
    assert isinstance(
        new_block.gradient("grad").gradient("grad_grad").values, np.ndarray
    )
    assert_equal(new_block.values, block.values.numpy())
    assert_equal(
        new_block.gradient("grad").values, block.gradient("grad").values.numpy()
    )
    assert_equal(
        new_block.gradient("grad").gradient("grad_grad").values,
        block.gradient("grad").gradient("grad_grad").values.numpy(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_numpy_to_torch_gpu_block(block):
    """Test a `block_to` conversion from numpy to a torch GPU tensor."""
    new_block = equistore.block_to(block, backend="torch", device="cuda")
    assert_equal(new_block.properties, block.properties)
    assert_equal(new_block.gradient("grad").samples, block.gradient("grad").samples)
    assert_equal(
        new_block.gradient("grad").gradient("grad_grad").components,
        block.gradient("grad").gradient("grad_grad").components,
    )
    assert isinstance(new_block.values, torch.Tensor)
    assert isinstance(new_block.gradient("grad").values, torch.Tensor)
    assert isinstance(
        new_block.gradient("grad").gradient("grad_grad").values, torch.Tensor
    )
    assert new_block.values.is_cuda
    assert new_block.gradient("grad").values.is_cuda
    assert new_block.gradient("grad").gradient("grad_grad").values.is_cuda
    assert_equal(new_block.values.cpu().numpy(), block.values)
    assert_equal(
        new_block.gradient("grad").values.cpu().numpy(), block.gradient("grad").values
    )
    assert_equal(
        new_block.gradient("grad").gradient("grad_grad").values.cpu().numpy(),
        block.gradient("grad").gradient("grad_grad").values,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_torch_to_gpu_block(block):
    """Test a `block_to` conversion from a torch CPU tensor to a torch GPU tensor."""
    block = equistore.block_to(block, backend="torch")
    new_block = equistore.block_to(block, device="cuda")
    assert_equal(new_block.properties, block.properties)
    assert_equal(new_block.gradient("grad").samples, block.gradient("grad").samples)
    assert_equal(
        new_block.gradient("grad").gradient("grad_grad").components,
        block.gradient("grad").gradient("grad_grad").components,
    )
    assert isinstance(new_block.values, torch.Tensor)
    assert isinstance(new_block.gradient("grad").values, torch.Tensor)
    assert isinstance(
        new_block.gradient("grad").gradient("grad_grad").values, torch.Tensor
    )
    assert new_block.values.is_cuda
    assert new_block.gradient("grad").values.is_cuda
    assert new_block.gradient("grad").gradient("grad_grad").values.is_cuda
    assert_equal(new_block.values.cpu().numpy(), block.values.numpy())
    assert_equal(
        new_block.gradient("grad").values.cpu().numpy(),
        block.gradient("grad").values.numpy(),
    )
    assert_equal(
        new_block.gradient("grad").gradient("grad_grad").values.cpu().numpy(),
        block.gradient("grad").gradient("grad_grad").values.numpy(),
    )


def test_change_dtype_block(block):
    """Test a `block_to` change of dtype"""
    new_block = equistore.block_to(block, dtype=np.float32)
    assert_equal(new_block.properties, block.properties)
    assert_equal(new_block.gradient("grad").samples, block.gradient("grad").samples)
    assert_equal(
        new_block.gradient("grad").gradient("grad_grad").components,
        block.gradient("grad").gradient("grad_grad").components,
    )
    assert isinstance(new_block.values, np.ndarray)
    assert isinstance(new_block.gradient("grad").values, np.ndarray)
    assert isinstance(
        new_block.gradient("grad").gradient("grad_grad").values, np.ndarray
    )
    assert new_block.values.dtype == np.float32
    assert new_block.gradient("grad").values.dtype == np.float32
    assert new_block.gradient("grad").gradient("grad_grad").values.dtype == np.float32
    assert np.allclose(new_block.values, block.values)
    assert np.allclose(
        new_block.gradient("grad").values,
        block.gradient("grad").values,
    )
    assert np.allclose(
        new_block.gradient("grad").gradient("grad_grad").values,
        block.gradient("grad").gradient("grad_grad").values,
    )


def test_wrong_arguments(tensor):
    """Test the `to` function with incorrect arguments."""
    with pytest.raises(TypeError, match="`tensor` should be an equistore `TensorMap`"):
        equistore.to(100)
    with pytest.raises(TypeError, match="`backend` should be passed as a `str`"):
        equistore.to(tensor, backend=10)
    with pytest.raises(
        ValueError,
        match="the `numpy` backend option does not support autograd gradient tracking",
    ):
        equistore.to(tensor, backend="numpy", requires_grad=True)
    with pytest.raises(ValueError, match="not supported"):
        equistore.to(tensor, backend="jax")


def test_numpy_to_torch(tensor):
    """Test a `to` conversion from numpy to torch."""
    new_tensor = equistore.to(tensor, backend="torch")
    assert equistore.equal_metadata(new_tensor, tensor)
    for _, new_block in new_tensor:
        assert isinstance(new_block.values, torch.Tensor)


def test_numpy_to_torch_switching_requires_grad(tensor):
    """Test a `to` conversion from numpy to torch, switching requires_grad on
    and off."""
    new_tensor = equistore.to(tensor, backend="torch")
    assert equistore.equal_metadata(new_tensor, tensor)
    for _, new_block in new_tensor:
        assert not new_block.values.requires_grad

    new_tensor = equistore.to(tensor, backend="torch", requires_grad=True)
    for _, new_block in new_tensor:
        assert new_block.values.requires_grad

    new_tensor = equistore.to(tensor, backend="torch", requires_grad=False)
    for _, new_block in new_tensor:
        assert not new_block.values.requires_grad


def test_torch_to_numpy(tensor):
    """Test a `to` conversion from torch to numpy."""
    tensor = equistore.to(tensor, backend="torch")
    new_tensor = equistore.to(tensor, backend="numpy")
    assert equistore.equal_metadata(new_tensor, tensor)
    for _, new_block in new_tensor:
        assert isinstance(new_block.values, np.ndarray)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_numpy_to_torch_gpu(tensor):
    """Test a `to` conversion from numpy to a torch GPU tensor."""
    new_tensor = equistore.to(tensor, backend="torch", device="cuda")
    assert equistore.equal_metadata(new_tensor, tensor)
    for _, new_block in new_tensor:
        assert isinstance(new_block.values, torch.Tensor)
        assert new_block.values.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_torch_to_gpu(tensor):
    """Test a `to` conversion from a torch CPU tensor to a torch GPU tensor."""
    tensor = equistore.to(tensor, backend="torch")
    new_tensor = equistore.to(tensor, device="cuda")
    assert equistore.equal_metadata(new_tensor, tensor)
    for _, new_block in new_tensor:
        assert isinstance(new_block.values, torch.Tensor)
        assert new_block.values.is_cuda


def test_change_dtype(tensor):
    """Test a `to` change of dtype"""
    new_tensor = equistore.to(tensor, dtype=np.float32)
    assert equistore.equal_metadata(new_tensor, tensor)
    for _, new_block in new_tensor:
        assert isinstance(new_block.values, np.ndarray)
        assert new_block.values.dtype == np.float32
