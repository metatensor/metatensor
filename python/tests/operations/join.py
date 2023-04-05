from os import path

import numpy as np
import pytest
from numpy.testing import assert_equal

import equistore
from equistore import Labels, TensorBlock, TensorMap


DATA_ROOT = path.join(path.dirname(__file__), "..", "data")


@pytest.fixture
def tensor():
    tensor = equistore.load(
        path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        use_numpy=True,
    )

    # Test if Tensormaps have at least one gradient. This avoids dropping gradient
    # tests silently by removing gradients from the reference data
    assert "positions" in tensor.block(0).gradients_list()

    return tensor


@pytest.fixture
def components_tensor():
    components_tensor = equistore.load(
        path.join(DATA_ROOT, "qm7-spherical-expansion.npz"), use_numpy=True
    )

    # Test if Tensormaps have at least one gradient. This avoids dropping gradient
    # tests silently by removing gradients from the reference data
    assert "positions" in components_tensor.block(0).gradients_list()

    return components_tensor


def test_wrong_axis(tensor):
    """Test error with unknown `axis` keyword."""
    with pytest.raises(ValueError, match="values for the `axis` parameter"):
        equistore.join([tensor, tensor, tensor], axis="foo")


def test_wrong_type(tensor):
    """Test if a wrong type (e.g., TensorMap) is provided."""
    with pytest.raises(TypeError, match="list or a tuple"):
        equistore.join(tensor, axis="properties")


@pytest.mark.parametrize("tensor", ([], ()))
def test_no_tensormaps(tensor):
    """Test if an empty list or tuple is provided."""
    with pytest.raises(ValueError, match="provide at least one"):
        equistore.join(tensor, axis="properties")


def test_single_tensormap(tensor):
    """Test if only one TensorMap is provided."""
    tensor_joined = equistore.join([tensor], axis="properties")
    assert tensor_joined is tensor


@pytest.mark.parametrize("axis", ["samples", "properties"])
def test_join_components(components_tensor, axis):
    """Test join for tensors with components."""
    equistore.join([components_tensor, components_tensor], axis=axis)


def test_join_properties_metadata(tensor):
    """Test join function with three tensormaps along `properties`.

    We check for the values below."""

    tensor_joined = equistore.join([tensor, tensor, tensor], axis="properties")

    # test property names
    names = tensor.block(0).properties.names
    assert tensor_joined.block(0).properties.names == ("tensor",) + names

    # test property values
    tensor_prop = np.unique(tensor_joined.block(0).properties["tensor"])
    assert set(tensor_prop) == set((0, 1, 2))

    # test if gradients exist
    assert sorted(tensor_joined[0].gradients_list()) == sorted(
        tensor[0].gradients_list()
    )


def test_join_properties_values(tensor):
    """Test values for joining along `properties`."""
    ts_1 = equistore.slice(tensor, axis="properties", labels=tensor[0].properties[:1])
    ts_2 = equistore.slice(tensor, axis="properties", labels=tensor[0].properties[1:])

    tensor_joined = equistore.join([ts_1, ts_2], axis="properties")

    # We can not use #equistore.equal_raise for the checks because the meta data
    # differs by the tensor entry.
    for i, block_tensor in tensor:
        block_tensor_joined = tensor_joined[i]

        assert_equal(block_tensor_joined.values, block_tensor.values)

        for parameter, gradient_tensor in block_tensor.gradients():
            gradient_tensor_joined = block_tensor_joined.gradient(parameter)
            assert_equal(gradient_tensor_joined.data, gradient_tensor.data)


def test_join_properties_with_same_property_names(tensor):
    """Test join function with three tensormaps along `properties`.

    We check for the values below."""

    tensor_joined = equistore.join([tensor, tensor, tensor], axis="properties")

    # test property names
    names = tensor.block(0).properties.names
    assert tensor_joined.block(0).properties.names == ("tensor",) + names

    # test property values
    tensor_prop = np.unique(tensor_joined.block(0).properties["tensor"])
    assert set(tensor_prop) == set((0, 1, 2))

    # test if gradients exist
    assert sorted(tensor_joined[0].gradients_list()) == sorted(
        tensor[0].gradients_list()
    )


def test_join_properties_with_different_property_names():
    """Test join function with tensormaps of different `property` names."""
    keys = Labels.arange("frame_a", 1)
    values = np.zeros([1, 1])
    samples = Labels.arange("idx", 1)

    tensor_map_a = TensorMap(
        keys=keys,
        blocks=[
            TensorBlock(
                values=values,
                samples=samples,
                components=[],
                properties=Labels.arange("prop1", 1),
            )
        ],
    )

    tensor_map_b = TensorMap(
        keys=keys,
        blocks=[
            TensorBlock(
                values=np.zeros([1, 1]),
                samples=samples,
                components=[],
                properties=Labels.arange("prop2", 1),
            )
        ],
    )

    tensor_joined = equistore.join([tensor_map_a, tensor_map_b], axis="properties")
    assert tensor_joined.property_names == ("tensor", "property")
    assert len(tensor_joined[0].properties) == 2


def test_join_samples_metadata(tensor):
    """Test join function with three tensormaps along `samples`."""
    tensor_joined = equistore.join([tensor, tensor, tensor], axis="samples")

    # test sample values
    assert len(tensor_joined.block(0).samples) == 3 * len(tensor.block(0).samples)

    # test if gradients exist
    assert sorted(tensor_joined[0].gradients_list()) == sorted(
        tensor[0].gradients_list()
    )


def test_join_samples_values(tensor):
    """Test values for joining along `samples`."""
    keys = Labels(
        names=tensor.keys.names,
        values=np.array(tensor.keys[0].tolist()).reshape(1, -1),
    )

    tm = TensorMap(keys, [tensor[0].copy()])
    ts_1 = equistore.slice(tm, axis="samples", labels=tensor[0].samples[:1])
    ts_2 = equistore.slice(tm, axis="samples", labels=tensor[0].samples[1:])

    tensor_joined = equistore.join([ts_1, ts_2], axis="samples")

    # We can not use #equistore.equal_raise for the checks because the meta data
    # differs by the tensor entry.
    for i, block_tensor in tm:
        block_tensor_joined = tensor_joined[i]

        assert_equal(block_tensor_joined.values, block_tensor.values)

        for parameter, gradient_tensor in block_tensor.gradients():
            gradient_tensor_joined = block_tensor_joined.gradient(parameter)
            assert_equal(gradient_tensor_joined.data, gradient_tensor.data)


def test_join_samples_with_different_sample_names():
    """Test join function raises an error with different `sample` names."""
    keys = Labels.arange("frame_a", 1)
    values = np.zeros([1, 1])
    properties = Labels.arange("idx", 1)

    tensor_map_a = TensorMap(
        keys=keys,
        blocks=[
            TensorBlock(
                values=values,
                samples=Labels.arange("samp1", 1),
                components=[],
                properties=properties,
            )
        ],
    )

    tensor_map_b = TensorMap(
        keys=keys,
        blocks=[
            TensorBlock(
                values=np.zeros([1, 1]),
                samples=Labels.arange("samp2", 1),
                components=[],
                properties=properties,
            )
        ],
    )

    with pytest.raises(ValueError, match="Sample names are not the same!"):
        equistore.join([tensor_map_a, tensor_map_b], axis="samples")
