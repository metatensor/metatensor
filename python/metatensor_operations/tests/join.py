import os

import numpy as np
import pytest
from numpy.testing import assert_equal


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


import metatensor as mts
from metatensor import Labels, TensorMap

from . import _gradcheck


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def tensor():
    tensor = mts.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.mts"))

    msg = (
        "Tensor must have at least one gradient. When no gradients are present certain "
        "tests will pass without testing anything."
    )
    assert len(tensor.block(0).gradients_list()) > 0, msg

    return tensor


@pytest.fixture
def components_tensor():
    components_tensor = mts.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.mts"))

    # Test if Tensormaps have at least one gradient. This avoids dropping gradient
    # tests silently by removing gradients from the reference data
    assert "positions" in components_tensor.block(0).gradients_list()

    return components_tensor


def test_wrong_axis(tensor):
    """Test error with unknown `axis` keyword."""
    with pytest.raises(ValueError, match="values for the `axis` parameter"):
        mts.join([tensor, tensor, tensor], axis="foo")


def test_wrong_type(tensor):
    """Test if a wrong type (e.g., TensorMap) is provided."""
    with pytest.raises(TypeError, match="list or a tuple"):
        mts.join(tensor, axis="properties")


def test_wrong_different_keys(tensor):
    """Test if a wrong type (e.g., TensorMap) is provided."""
    match = "'foo' is not a valid option for `different_keys`"
    with pytest.raises(ValueError, match=match):
        mts.join([tensor, tensor], axis="properties", different_keys="foo")


@pytest.mark.parametrize("tensor", ([], ()))
def test_no_tensormaps(tensor):
    """Test if an empty list or tuple is provided."""
    with pytest.raises(ValueError, match="provide at least one"):
        mts.join(tensor, axis="properties")


def test_single_tensormap(tensor):
    """Test if only one TensorMap is provided."""
    joined_tensor = mts.join([tensor], axis="properties")
    assert joined_tensor is tensor


@pytest.mark.parametrize("axis", ["samples", "properties"])
def test_join_components(components_tensor, axis):
    """Test join for tensors with components."""
    mts.join([components_tensor, components_tensor], axis=axis, add_dimension="tensor")


def test_join_properties_metadata(tensor):
    """Test join function with three tensors along `properties`.

    We check for the values below"""

    joined_tensor = mts.join(
        [tensor, tensor, tensor], axis="properties", add_dimension="tensor"
    )

    # test property names
    names = tensor.block(0).properties.names
    assert joined_tensor.block(0).properties.names == names + ["tensor"]

    # test property values
    tensor_prop = np.unique(joined_tensor.block(0).properties["tensor"])
    assert set(tensor_prop) == set((0, 1, 2))

    # test if gradients exist
    assert sorted(joined_tensor[0].gradients_list()) == sorted(
        tensor[0].gradients_list()
    )


def test_join_properties_values(tensor):
    """Test values for joining along `properties`"""
    first_property = Labels(
        tensor[0].properties.names,
        tensor[0].properties.values[:1],
    )
    slice_1 = mts.slice(tensor, axis="properties", selection=first_property)

    other_properties = Labels(
        tensor[0].properties.names,
        tensor[0].properties.values[1:],
    )
    slice_2 = mts.slice(tensor, axis="properties", selection=other_properties)

    joined_tensor = mts.join([slice_1, slice_2], axis="properties")

    # We can not use `mts.equal_raise` for the checks because the meta data
    # differs by the tensor entry.
    for key, block in tensor.items():
        joined_block = joined_tensor[key]

        assert_equal(joined_block.values, block.values)

        for parameter, gradient in block.gradients():
            joined_gradient = joined_block.gradient(parameter)
            assert_equal(joined_gradient.values, gradient.values)


def test_join_properties_with_same_property_names(tensor):
    """Test join function with three tensor along properties"""

    joined_tensor = mts.join(
        [tensor, tensor, tensor], axis="properties", add_dimension="added"
    )

    # test property names
    names = tensor.block(0).properties.names
    assert joined_tensor.block(0).properties.names == names + ["added"]

    # test property values
    added = np.unique(joined_tensor.block(0).properties["added"])
    assert set(added) == set((0, 1, 2))

    # test if gradients exist
    assert sorted(joined_tensor[0].gradients_list()) == sorted(
        tensor[0].gradients_list()
    )


def test_join_properties_with_different_property_names():
    """Test join function with tensors of different property names"""

    tensor_a = TensorMap(
        keys=Labels.range("_", 1),
        blocks=[mts.block_from_array(np.zeros([1, 1]), property_names=["p_1"])],
    )

    tensor_b = TensorMap(
        keys=Labels.range("_", 1),
        blocks=[mts.block_from_array(np.zeros([1, 1]), property_names=["p_2"])],
    )

    joined_tensor = mts.join([tensor_a, tensor_b], axis="properties")
    assert joined_tensor.property_names == ["joined_index", "property"]
    assert len(joined_tensor[0].properties) == 2


def test_join_samples_metadata(tensor):
    """Test join function with three tensors along `samples`"""
    joined_tensor = mts.join(
        [tensor, tensor, tensor], axis="samples", add_dimension="tensor"
    )

    # test sample values
    assert len(joined_tensor.block(0).samples) == 3 * len(tensor.block(0).samples)

    # test if gradients exist
    assert sorted(joined_tensor[0].gradients_list()) == sorted(
        tensor[0].gradients_list()
    )


def test_join_samples_values(tensor):
    """Test values for joining along `samples`"""
    keys = Labels(
        names=tensor.keys.names,
        values=tensor.keys.values[:1],
    )

    tensor = TensorMap(keys, [tensor[0].copy()])
    first_samples = Labels(
        tensor[0].samples.names,
        tensor[0].samples.values[:1],
    )
    slice_1 = mts.slice(tensor, axis="samples", selection=first_samples)

    other_samples = Labels(
        tensor[0].samples.names,
        tensor[0].samples.values[1:],
    )
    slice_2 = mts.slice(tensor, axis="samples", selection=other_samples)

    joined_tensor = mts.join([slice_1, slice_2], axis="samples")

    # We can not use #mts.equal_raise for the checks because the meta data
    # differs by the tensor entry.
    for key, block in tensor.items():
        joined_block = joined_tensor[key]

        assert_equal(joined_block.values, block.values)

        for parameter, gradient in block.gradients():
            joined_gradient = joined_block.gradient(parameter)
            assert_equal(joined_gradient.values, gradient.values)


def test_join_samples_with_different_sample_names():
    """Test join function raises an error with different sample names"""
    tensor_a = TensorMap(
        keys=Labels.range("_", 1),
        blocks=[mts.block_from_array(np.zeros([1, 1]), sample_names=["s_1"])],
    )

    tensor_b = TensorMap(
        keys=Labels.range("_", 1),
        blocks=[mts.block_from_array(np.zeros([1, 1]), sample_names=["s_2"])],
    )

    message = (
        "Different tensor have different sample names in `join`. "
        "Joining along samples with different sample names will lose "
        "information and is not supported."
    )
    with pytest.raises(ValueError, match=message):
        mts.join([tensor_a, tensor_b], axis="samples")


def test_split_join_samples(tensor):
    """Test if split and joining along `samples` results in the same TensorMap."""

    labels_1 = Labels(names=["system"], values=np.arange(4).reshape(-1, 1))
    labels_2 = Labels(names=["system"], values=np.arange(4, 10).reshape(-1, 1))

    split_tensors = mts.split(
        tensor=tensor, axis="samples", selections=[labels_1, labels_2]
    )
    joined_tensor = mts.join(split_tensors, axis="samples")

    assert joined_tensor == tensor


def test_split_join_properties(tensor):
    """Test if split and joining along `properties` results in the same TensorMap."""
    properties = tensor[0].properties

    labels_1 = Labels(names=properties.names, values=properties.values[:5])
    labels_2 = Labels(names=properties.names, values=properties.values[5:])

    split_tensors = mts.split(
        tensor=tensor, axis="properties", selections=[labels_1, labels_2]
    )
    joined_tensor = mts.join(split_tensors, axis="properties")

    assert joined_tensor == tensor


@pytest.mark.parametrize("axis", ["samples", "properties"])
def test_intersection_join(axis, components_tensor):
    tensor_1 = components_tensor

    labels_remove = Labels(names=tensor_1.keys.names, values=tensor_1.keys.values[:1])
    labels_present = Labels(names=tensor_1.keys.names, values=tensor_1.keys.values[1:])

    tensor_2 = mts.drop_blocks(components_tensor, labels_remove, copy=True)

    joined_tensor = mts.join(
        [tensor_1, tensor_2],
        axis=axis,
        different_keys="intersection",
        add_dimension="tensor",
    )

    assert joined_tensor.keys == labels_present


@pytest.mark.parametrize("axis", ["samples", "properties"])
def test_union_join(axis, components_tensor):
    tensor_1 = components_tensor

    labels_remove = Labels(names=tensor_1.keys.names, values=tensor_1.keys.values[:1])

    tensor_2 = mts.drop_blocks(components_tensor, labels_remove, copy=True)

    joined_tensor = mts.join(
        [tensor_1, tensor_2], axis=axis, different_keys="union", add_dimension="tensor"
    )

    assert joined_tensor.keys == tensor_1.keys

    # First block should be same as in tensor_1
    assert joined_tensor[0].values.shape == tensor_1[0].values.shape

    # All other blocks should have doubled the number of samples/features
    for i_block in range(1, len(tensor_1.keys)):
        ref_block = tensor_1[i_block]
        joined_block = joined_tensor[i_block]

        # Check values
        if axis == "samples":
            ref_shape = (2 * ref_block.values.shape[0],) + ref_block.values.shape[1:]
        else:
            ref_shape = ref_block.values.shape[:-1] + (2 * ref_block.values.shape[-1],)

        assert joined_block.values.shape == ref_shape

        # Check values
        for parameter, joined_gradient in joined_block.gradients():
            ref_gradient = ref_block.gradient(parameter)

            if axis == "samples":
                ref_gradient_shape = (
                    2 * ref_gradient.values.shape[0],
                ) + ref_gradient.values.shape[1:]
            else:
                ref_gradient_shape = ref_gradient.values.shape[:-1] + (
                    2 * ref_gradient.values.shape[-1],
                )

            assert joined_gradient.values.shape == ref_gradient_shape


def test_finite_difference_sample_join():
    def function(array):
        tensor_1 = _gradcheck.tensor_with_grad_a(array, parameter="g")
        tensor_2 = _gradcheck.tensor_with_grad_b(array, parameter="g")
        return mts.join([tensor_1, tensor_2], axis="samples", add_dimension="t")

    rng = np.random.default_rng(seed=123456)
    array = rng.random((42, 3))
    _gradcheck.check_finite_differences(function, array, parameter="g")

    if HAS_TORCH:
        rng = torch.Generator()
        rng.manual_seed(123456)
        array = torch.rand(55, 3, dtype=torch.float64, generator=rng)
        _gradcheck.check_finite_differences(function, array, parameter="g")


def test_finite_difference_property_join():
    def function(array):
        tensor_1 = _gradcheck.tensor_with_grad_a(array, parameter="g")
        tensor_2 = _gradcheck.tensor_with_grad_b(array, parameter="g")
        return mts.join([tensor_1, tensor_2], axis="properties", add_dimension="t")

    rng = np.random.default_rng(seed=123456)
    array = rng.random((42, 3))
    _gradcheck.check_finite_differences(function, array, parameter="g")

    if HAS_TORCH:
        rng = torch.Generator()
        rng.manual_seed(123456)
        array = torch.rand(55, 3, dtype=torch.float64, generator=rng)
        _gradcheck.check_finite_differences(function, array, parameter="g")
