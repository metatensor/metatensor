import os

import numpy as np
import pytest
from numpy.testing import assert_equal

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def tensor():
    return metatensor.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"))


@pytest.fixture
def tensor_extra():
    """TensorMap with gradient containing extra columns in keys, samples, properties"""
    block = TensorBlock(
        values=np.array([[42]]),
        samples=Labels(["sample", "extra"], np.array([[0, 0]])),
        components=[],
        properties=Labels(["properties", "extra"], np.array([[0, 0]])),
    )

    block.add_gradient("gradient", block.copy())

    keys = Labels(["keys", "extra"], np.array([[0, 0]]))
    return TensorMap(keys=keys, blocks=[block])


def test_append_keys(tensor):
    values = np.arange(17)
    new_tensor = metatensor.append_dimension(
        tensor,
        axis="keys",
        name="foo",
        values=values,
    )

    assert new_tensor.keys.names[-1] == "foo"
    assert_equal(new_tensor.keys.values[:, -1], values)


def test_append_samples(tensor_extra):
    values = np.array([42])
    new_tensor = metatensor.append_dimension(
        tensor_extra,
        axis="samples",
        name="foo",
        values=values,
    )

    assert new_tensor.sample_names[-1] == "foo"

    for block in new_tensor:
        assert block.samples.values[:, -1] == values


def test_append_properties(tensor):
    values = np.arange(80)
    new_tensor = metatensor.append_dimension(
        tensor,
        axis="properties",
        name="foo",
        values=values,
    )

    assert new_tensor.property_names[-1] == "foo"

    for block in new_tensor:
        assert_equal(block.properties.values[:, -1], values)

        for _, gradient in block.gradients():
            assert gradient.properties.names[-1] == "foo"
            assert_equal(gradient.properties.values[:, -1], values)


def test_append_unknown_axis(tensor):
    with pytest.raises(ValueError, match="'foo' is not a valid axis."):
        metatensor.append_dimension(tensor, axis="foo", name="foo", values=10)


def test_insert_keys(tensor):
    values = np.arange(17)
    index = 0
    new_tensor = metatensor.insert_dimension(
        tensor,
        axis="keys",
        index=index,
        name="foo",
        values=values,
    )

    assert new_tensor.keys.names[index] == "foo"
    assert_equal(new_tensor.keys.values[:, index], values)


def test_insert_samples(tensor_extra):
    values = np.array([42])
    index = 0
    new_tensor = metatensor.insert_dimension(
        tensor_extra,
        axis="samples",
        index=index,
        name="foo",
        values=values,
    )

    assert new_tensor.sample_names[index] == "foo"

    for block in new_tensor:
        assert block.samples.values[:, index] == values


def test_insert_properties(tensor):
    values = np.arange(80)
    index = 0
    new_tensor = metatensor.insert_dimension(
        tensor,
        axis="properties",
        index=index,
        name="foo",
        values=values,
    )

    assert new_tensor.property_names[index] == "foo"

    for block in new_tensor:
        assert_equal(block.properties.values[:, index], values)

        for _, gradient in block.gradients():
            assert gradient.properties.names[index] == "foo"
            assert_equal(gradient.properties.values[:, index], np.arange(80))


def test_insert_unknown_axis(tensor):
    with pytest.raises(ValueError, match="'foo' is not a valid axis."):
        metatensor.insert_dimension(tensor, axis="foo", index=0, name="foo", values=10)


def test_permute_keys(tensor):
    dimensions_indexes = [2, 0, 1]
    new_tensor = metatensor.permute_dimensions(
        tensor, axis="keys", dimensions_indexes=dimensions_indexes
    )

    assert new_tensor.keys.names == [
        "neighbor_2_type",
        "center_type",
        "neighbor_1_type",
    ]
    assert_equal(new_tensor.keys.values, tensor.keys.values[:, dimensions_indexes])


def test_permute_samples(tensor):
    dimensions_indexes = [1, 0]
    new_tensor = metatensor.permute_dimensions(
        tensor, axis="samples", dimensions_indexes=dimensions_indexes
    )

    assert new_tensor.sample_names == ["atom", "system"]

    for key, block in new_tensor.items():
        ref_block = tensor.block(key)
        assert_equal(
            block.samples.values, ref_block.samples.values[:, dimensions_indexes]
        )
        for parameter, gradient in block.gradients():
            if parameter == "positions":
                gradient_sample_names = ["sample", "system", "atom"]
            elif parameter == "strain":
                gradient_sample_names = ["sample"]
            assert gradient.samples.names == gradient_sample_names


def test_permute_properties(tensor):
    dimensions_indexes = [2, 0, 1]
    new_tensor = metatensor.permute_dimensions(
        tensor, axis="properties", dimensions_indexes=dimensions_indexes
    )

    assert new_tensor.property_names == ["n_2", "l", "n_1"]

    for key, block in new_tensor.items():
        ref_block = tensor.block(key)
        assert_equal(
            block.properties.values, ref_block.properties.values[:, dimensions_indexes]
        )


def test_permute_unknown_axis(tensor):
    with pytest.raises(ValueError, match="'foo' is not a valid axis."):
        metatensor.permute_dimensions(tensor, axis="foo", dimensions_indexes=[1])


def test_rename_keys(tensor):
    old = tensor.keys.names[0]

    new_tensor = metatensor.rename_dimension(tensor, axis="keys", old=old, new="foo")
    assert new_tensor.keys.names[0] == "foo"


def test_rename_samples(tensor):
    new_tensor = metatensor.rename_dimension(
        tensor,
        axis="samples",
        old="system",
        new="foo",
    )
    assert new_tensor.sample_names[0] == "foo"

    for block in new_tensor:
        for parameter, gradient in block.gradients():
            if parameter == "positions":
                gradient_sample_names = ["sample", "foo", "atom"]
            elif parameter == "strain":
                gradient_sample_names = ["sample"]
            assert gradient.samples.names == gradient_sample_names


def test_rename_properties(tensor):
    old = tensor.property_names[0]

    new_tensor = metatensor.rename_dimension(
        tensor,
        axis="properties",
        old=old,
        new="foo",
    )
    assert new_tensor.property_names[0] == "foo"

    for block in new_tensor:
        for _, gradient in block.gradients():
            assert gradient.properties.names[0] == "foo"


def test_rename_unknown_axis(tensor):
    with pytest.raises(ValueError, match="'foo' is not a valid axis."):
        metatensor.rename_dimension(tensor, axis="foo", old="foo", new="foo")


def test_remove_keys(tensor_extra):
    new_tensor = metatensor.remove_dimension(tensor_extra, axis="keys", name="extra")
    assert new_tensor.keys.names == ["keys"]


def test_remove_samples(tensor_extra):
    new_tensor = metatensor.remove_dimension(
        tensor_extra,
        axis="samples",
        name="extra",
    )
    assert new_tensor.sample_names == ["sample"]


def test_remove_properties(tensor_extra):
    new_tensor = metatensor.remove_dimension(
        tensor_extra,
        axis="properties",
        name="extra",
    )
    assert new_tensor.property_names == ["properties"]

    for block in new_tensor:
        for _, gradient in block.gradients():
            assert gradient.properties.names == ["properties"]


def test_remove_unknown_axis(tensor):
    with pytest.raises(ValueError, match="'foo' is not a valid axis."):
        metatensor.remove_dimension(tensor, axis="foo", name="foo")


def test_not_unique_after(tensor):
    """Test error raise if the the labels after the removal would be not valid."""
    match = (
        r"invalid parameter: can not have the same label value multiple time: \[1, 1\] "
        r"is already present at position 0"
    )
    with pytest.raises(metatensor.status.MetatensorError, match=match):
        metatensor.remove_dimension(tensor, axis="keys", name="center_type")


def test_insert_dimension_wrong_size(tensor):
    """
    Tests that passing an int for the `values` is ok, but an array of the
    wrong size isn't
    """

    message = "the new `values` contains 1 entries, but the Labels contains 52"
    with pytest.raises(ValueError, match=message):
        metatensor.insert_dimension(
            tensor,
            axis="samples",
            name="new_dimension",
            values=np.array([0]),
            index=0,
        )

    metatensor.insert_dimension(
        tensor,
        axis="samples",
        name="new_dimension",
        values=0,
        index=0,
    )
