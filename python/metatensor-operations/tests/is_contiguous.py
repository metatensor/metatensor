import os

import pytest

import metatensor
from metatensor import TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def tensor():
    """Loads a TensorMap from file for use in tests"""
    return metatensor.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        use_numpy=True,
    )


def _non_contiguous_block(block: TensorBlock) -> TensorBlock:
    """
    Make a non-contiguous block by reversing the order in both the main value block and
    the gradient block(s).
    """
    new_vals = block.values.copy()[::-1, ::-1]
    new_block = TensorBlock(
        values=new_vals,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )
    for param, gradient in block.gradients():
        new_grad = gradient.values.copy()[::-1, ::-1]
        new_gradient = TensorBlock(
            values=new_grad,
            samples=gradient.samples,
            components=gradient.components,
            properties=gradient.properties,
        )
        new_block.add_gradient(param, new_gradient)

    return new_block


@pytest.fixture
def non_contiguous_tensor(tensor) -> TensorMap:
    """
    Make a TensorMap non-contiguous by reversing the order of the samples/properties
    rows/columns in all values and gradient blocks.
    """
    keys = tensor.keys
    new_blocks = []

    for block in tensor.blocks():
        new_block = _non_contiguous_block(block)
        new_blocks.append(new_block)

    return TensorMap(keys=keys, blocks=new_blocks)


def test_is_contiguous_block(tensor):
    assert metatensor.is_contiguous_block(tensor.block(0))
    assert not metatensor.is_contiguous_block(_non_contiguous_block(tensor.block(0)))


def test_is_contiguous(tensor, non_contiguous_tensor):
    assert metatensor.is_contiguous(tensor)
    assert not metatensor.is_contiguous(non_contiguous_tensor)
