import numpy as np
import pytest
import torch

import metatensor
from metatensor import Labels, TensorMap


@pytest.fixture
def block():
    # Returns a block with nested gradients
    values = np.arange(42, dtype=np.float64).reshape(7, 3, 2)
    grad_values = np.arange(42, dtype=np.float64).reshape(7, 3, 2)
    grad_grad_values = np.arange(42, dtype=np.float64).reshape(7, 3, 2)
    block = metatensor.block_from_array(values)
    grad_block = metatensor.block_from_array(grad_values)
    grad_grad_block = metatensor.block_from_array(grad_grad_values)
    grad_block.add_gradient("grad_grad", grad_grad_block)
    block.add_gradient("grad", grad_block)
    return block


@pytest.fixture
def tensor():
    # Returns a TensorMap with two blocks
    values_1 = np.arange(42, dtype=np.float64).reshape(7, 3, 2)
    block_1 = metatensor.block_from_array(values_1)
    values_2 = np.arange(100, dtype=np.float64).reshape(10, 5, 2)
    block_2 = metatensor.block_from_array(values_2)
    tensor = TensorMap(keys=Labels.range("dummy", 2), blocks=[block_1, block_2])
    return tensor


def test_wrong_arguments_block(block):
    """Test the `block_to` function with incorrect arguments."""
    with pytest.raises(TypeError, match="`block` should be a metatensor `TensorBlock`"):
        metatensor.block_to(100)

    with pytest.raises(TypeError, match="'backend' should be given as a string"):
        metatensor.block_to(block, backend=10)

    message = "the `numpy` backend option does not support autograd gradient tracking"
    with pytest.raises(ValueError, match=message):
        metatensor.block_to(block, backend="numpy", requires_grad=True)

    with pytest.raises(ValueError, match="backend 'jax' is not supported"):
        metatensor.block_to(block, backend="jax")


def test_numpy_to_torch_block(block):
    """Test a `block_to` conversion from numpy to torch."""
    new_block = metatensor.block_to(block, backend="torch")

    assert isinstance(new_block.values, torch.Tensor)
    assert metatensor.equal_metadata_block(new_block, block)
    np.testing.assert_equal(new_block.values.numpy(), block.values)

    for parameter, gradient in block.gradients():
        new_gradient = new_block.gradient(parameter)

        assert isinstance(new_gradient.values, torch.Tensor)
        assert metatensor.equal_metadata_block(new_gradient, gradient)
        np.testing.assert_equal(new_gradient.values.numpy(), gradient.values)

        for parameter_2, gradient_gradient in gradient.gradients():
            new_gradient_gradient = new_gradient.gradient(parameter_2)

            assert isinstance(new_gradient_gradient.values, torch.Tensor)
            assert metatensor.equal_metadata_block(
                new_gradient_gradient, gradient_gradient
            )
            np.testing.assert_equal(
                new_gradient_gradient.values.numpy(), gradient_gradient.values
            )


def test_torch_to_numpy_block(block):
    """Test a `block_to` conversion from torch to numpy."""
    block = metatensor.block_to(block, backend="torch")
    new_block = metatensor.block_to(block, backend="numpy")

    assert isinstance(block.values, torch.Tensor)
    assert isinstance(new_block.values, np.ndarray)
    assert metatensor.equal_metadata_block(new_block, block)
    np.testing.assert_equal(new_block.values, block.values.numpy())

    for parameter, gradient in block.gradients():
        new_gradient = new_block.gradient(parameter)

        assert isinstance(gradient.values, torch.Tensor)
        assert isinstance(new_gradient.values, np.ndarray)
        assert metatensor.equal_metadata_block(new_gradient, gradient)
        np.testing.assert_equal(new_gradient.values, gradient.values.numpy())

        for parameter_2, gradient_gradient in gradient.gradients():
            new_gradient_gradient = new_gradient.gradient(parameter_2)

            assert isinstance(gradient_gradient.values, torch.Tensor)
            assert isinstance(new_gradient_gradient.values, np.ndarray)
            assert metatensor.equal_metadata_block(
                new_gradient_gradient, gradient_gradient
            )
            np.testing.assert_equal(
                new_gradient_gradient.values, gradient_gradient.values.numpy()
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_numpy_to_torch_gpu_block(block):
    """Test a `block_to` conversion from numpy to a torch GPU tensor."""
    new_block = metatensor.block_to(block, backend="torch", device="cuda")

    assert isinstance(new_block.values, torch.Tensor)
    assert metatensor.equal_metadata_block(new_block, block)
    assert new_block.values.device.type == "cuda"
    np.testing.assert_equal(new_block.values.cpu().numpy(), block.values)

    for parameter, gradient in block.gradients():
        new_gradient = new_block.gradient(parameter)

        assert isinstance(new_gradient.values, torch.Tensor)
        assert metatensor.equal_metadata_block(new_gradient, gradient)
        np.testing.assert_equal(new_gradient.values.cpu().numpy(), gradient.values)

        for parameter_2, gradient_gradient in gradient.gradients():
            new_gradient_gradient = new_gradient.gradient(parameter_2)

            assert isinstance(new_gradient_gradient.values, torch.Tensor)
            assert metatensor.equal_metadata_block(
                new_gradient_gradient, gradient_gradient
            )
            np.testing.assert_equal(
                new_gradient_gradient.values.cpu().numpy(), gradient_gradient.values
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_torch_to_gpu_block(block):
    """Test a `block_to` conversion from a torch CPU tensor to a torch GPU tensor."""
    block = metatensor.block_to(block, backend="torch")
    new_block = metatensor.block_to(block, device="cuda")

    assert isinstance(new_block.values, torch.Tensor)
    assert metatensor.equal_metadata_block(new_block, block)
    np.testing.assert_equal(new_block.values.cpu().numpy(), block.values.numpy())
    assert new_block.values.device.type == "cuda"
    assert block.values.device.type == "cpu"

    for parameter, gradient in block.gradients():
        new_gradient = new_block.gradient(parameter)

        assert isinstance(new_gradient.values, torch.Tensor)
        assert metatensor.equal_metadata_block(new_gradient, gradient)
        np.testing.assert_equal(
            new_gradient.values.cpu().numpy(), gradient.values.numpy()
        )
        assert new_gradient.values.device.type == "cuda"
        assert gradient.values.device.type == "cpu"

        for parameter_2, gradient_gradient in gradient.gradients():
            new_gradient_gradient = new_gradient.gradient(parameter_2)

            assert isinstance(new_gradient_gradient.values, torch.Tensor)
            assert metatensor.equal_metadata_block(
                new_gradient_gradient, gradient_gradient
            )
            np.testing.assert_equal(
                new_gradient_gradient.values.cpu().numpy(),
                gradient_gradient.values.numpy(),
            )
            assert new_gradient_gradient.values.device.type == "cuda"
            assert gradient_gradient.values.device.type == "cpu"


def test_change_dtype_block(block):
    """Test a `block_to` change of dtype"""
    new_block = metatensor.block_to(block, dtype=np.float32)

    assert metatensor.equal_metadata_block(new_block, block)
    assert np.allclose(new_block.values, block.values)

    assert block.values.dtype == np.float64
    assert new_block.values.dtype == np.float32

    for parameter, gradient in block.gradients():
        new_gradient = new_block.gradient(parameter)

        assert metatensor.equal_metadata_block(new_gradient, gradient)
        assert np.allclose(new_gradient.values, gradient.values)

        assert gradient.values.dtype == np.float64
        assert new_gradient.values.dtype == np.float32

        for parameter_2, gradient_gradient in gradient.gradients():
            new_gradient_gradient = new_gradient.gradient(parameter_2)

            assert metatensor.equal_metadata_block(
                new_gradient_gradient, gradient_gradient
            )
            assert np.allclose(new_gradient_gradient.values, gradient_gradient.values)

            assert gradient_gradient.values.dtype == np.float64
            assert new_gradient_gradient.values.dtype == np.float32


def test_wrong_arguments(tensor):
    """Test the `to` function with incorrect arguments."""
    with pytest.raises(TypeError, match="`tensor` should be a metatensor `TensorMap`"):
        metatensor.to(100)

    with pytest.raises(TypeError, match="'backend' should be given as a string"):
        metatensor.to(tensor, backend=10)

    message = "the `numpy` backend option does not support autograd gradient tracking"
    with pytest.raises(ValueError, match=message):
        metatensor.to(tensor, backend="numpy", requires_grad=True)

    with pytest.raises(ValueError, match="backend 'jax' is not supported"):
        metatensor.to(tensor, backend="jax")


def test_numpy_to_torch(tensor):
    """Test a `to` conversion from numpy to torch."""
    new_tensor = metatensor.to(tensor, backend="torch")
    assert metatensor.equal_metadata(new_tensor, tensor)
    for new_block in new_tensor:
        assert isinstance(new_block.values, torch.Tensor)


def test_numpy_to_torch_switching_requires_grad(tensor):
    """
    Test a `to` conversion from numpy to torch, switching requires_grad on and off.
    """
    new_tensor = metatensor.to(tensor, backend="torch")
    assert metatensor.equal_metadata(new_tensor, tensor)
    for new_block in new_tensor:
        assert not new_block.values.requires_grad

    new_tensor = metatensor.to(tensor, backend="torch", requires_grad=True)
    for new_block in new_tensor:
        assert new_block.values.requires_grad

    new_tensor = metatensor.to(tensor, backend="torch", requires_grad=False)
    for new_block in new_tensor:
        assert not new_block.values.requires_grad


def test_torch_to_numpy(tensor):
    """Test a `to` conversion from torch to numpy."""
    tensor = metatensor.to(tensor, backend="torch")
    new_tensor = metatensor.to(tensor, backend="numpy")
    assert metatensor.equal_metadata(new_tensor, tensor)
    for new_block in new_tensor:
        assert isinstance(new_block.values, np.ndarray)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_numpy_to_torch_gpu(tensor):
    """Test a `to` conversion from numpy to a torch GPU tensor."""
    new_tensor = metatensor.to(tensor, backend="torch", device="cuda")
    assert metatensor.equal_metadata(new_tensor, tensor)
    for new_block in new_tensor:
        assert isinstance(new_block.values, torch.Tensor)
        assert new_block.values.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_torch_to_gpu(tensor):
    """Test a `to` conversion from a torch CPU tensor to a torch GPU tensor."""
    tensor = metatensor.to(tensor, backend="torch")
    new_tensor = metatensor.to(tensor, device="cuda")
    assert metatensor.equal_metadata(new_tensor, tensor)
    for new_block in new_tensor:
        assert isinstance(new_block.values, torch.Tensor)
        assert new_block.values.is_cuda


def test_change_dtype(tensor):
    """Test a `to` change of dtype"""
    new_tensor = metatensor.to(tensor, dtype=np.float32)
    assert metatensor.equal_metadata(new_tensor, tensor)
    for new_block in new_tensor:
        assert isinstance(new_block.values, np.ndarray)
        assert new_block.values.dtype == np.float32
