import numpy as np
import pytest

import metatensor


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.parametrize("n_axes", [0, 1])
def test_too_few_axes(n_axes):
    """Test block_from_array when too few axes are provided."""
    with pytest.raises(ValueError, match="at least"):
        metatensor.block_from_array(np.zeros((4,) * n_axes))


def test_without_components():
    """Test block_from_array for a 2D array."""
    array = np.zeros((6, 7))
    block = metatensor.block_from_array(array)
    assert block.values is array

    assert block.samples.names == ["sample"]
    np.testing.assert_equal(
        block.samples.values, np.arange(array.shape[0]).reshape((-1, 1))
    )

    assert block.properties.names == ["property"]
    np.testing.assert_equal(
        block.properties.values, np.arange(array.shape[1]).reshape((-1, 1))
    )


def test_with_components():
    """Test block_from_array with components."""
    array = array = np.zeros((6, 5, 7))
    block = metatensor.block_from_array(array)
    assert block.values is array

    assert block.samples.names == ["sample"]
    np.testing.assert_equal(
        block.samples.values, np.arange(array.shape[0]).reshape((-1, 1))
    )

    assert len(block.components) == 1
    component = block.components[0]
    assert component.names == ["component_1"]
    np.testing.assert_equal(
        component.values, np.arange(array.shape[1]).reshape((-1, 1))
    )

    assert block.properties.names == ["property"]
    np.testing.assert_equal(
        block.properties.values, np.arange(array.shape[2]).reshape((-1, 1))
    )


@pytest.mark.parametrize("sample_names", [None, ["a"], ["a", "b"]])
@pytest.mark.parametrize("property_names", [None, ["A"], ["A", "B"]])
@pytest.mark.parametrize("component_names", [None, ["c1", "c2", "c3"]])
def test_with_label_names(sample_names, component_names, property_names):
    """Test block_from_array with explicit sample and property names."""
    array = array = np.zeros((3, 2, 1, 2, 3))

    if sample_names is None:
        sample_names = ["sample"]
    if property_names is None:
        property_names = ["property"]

    if component_names is not None and (
        len(sample_names) != 1 or len(property_names) != 1
    ):
        with pytest.raises(ValueError, match=".*does not have enough dimensions.*"):
            block = metatensor.block_from_array(
                array,
                sample_names=sample_names,
                component_names=component_names,
                property_names=property_names,
            )
        return
    else:
        block = metatensor.block_from_array(
            array,
            sample_names=sample_names,
            component_names=component_names,
            property_names=property_names,
        )

    expected_shape = (
        (
            np.prod(
                array.shape[: len(sample_names)],
            ),
        )
        + tuple(
            array.shape[i]
            for i in range(len(sample_names), len(array.shape) - len(property_names))
        )
        + (
            np.prod(
                array.shape[-len(property_names) :],
            ),
        )
    )

    for actual_shape, expected in zip(block.values.shape, expected_shape):
        assert actual_shape == expected

    assert len(block.samples.names) == len(sample_names)
    for actual_shape, expected in zip(block.samples.names, sample_names):
        assert actual_shape == expected

    assert len(block.properties.names) == len(property_names)
    for actual_shape, expected in zip(block.properties.names, property_names):
        assert actual_shape == expected

    assert len(block.components) == 5 - len(sample_names) - len(property_names)


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_with_components():
    """Test block_from_array with components and torch arrays"""
    array = array = torch.zeros((6, 5, 7))
    block = metatensor.block_from_array(array)
    assert block.values is array

    assert block.samples.names == ["sample"]
    np.testing.assert_equal(
        block.samples.values, np.arange(array.shape[0]).reshape((-1, 1))
    )

    assert len(block.components) == 1
    component = block.components[0]
    assert component.names == ["component_1"]
    np.testing.assert_equal(
        component.values, np.arange(array.shape[1]).reshape((-1, 1))
    )

    assert block.properties.names == ["property"]
    np.testing.assert_equal(
        block.properties.values, np.arange(array.shape[2]).reshape((-1, 1))
    )
