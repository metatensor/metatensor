from os import path

import numpy as np
import pytest
from numpy.testing import assert_equal

import equistore
from equistore import Labels, TensorMap
from equistore.status import EquistoreError


DATA_ROOT = path.join(path.dirname(__file__), "..", "data")


class TestJoinTensorMap:
    @pytest.fixture
    def tensor(self):
        tensor = equistore.load(
            path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            use_numpy=True,
        )

        # Test if Tensormaps have at least one gradient. This avoids dropping gradient
        # tests silently by removing gradients from the reference data
        assert "positions" in tensor.block(0).gradients_list()

        return tensor

    @pytest.fixture
    def components_tensor(self):
        components_tensor = equistore.load(
            path.join(DATA_ROOT, "qm7-spherical-expansion.npz"), use_numpy=True
        )

        # Test if Tensormaps have at least one gradient. This avoids dropping gradient
        # tests silently by removing gradients from the reference data
        assert "positions" in components_tensor.block(0).gradients_list()

        return components_tensor

    @pytest.fixture
    def tensor_single_block(self, tensor):
        keys_first_block = Labels(
            names=tensor.keys.names,
            values=np.array(tensor.keys[0].tolist()).reshape(1, -1),
        )
        return TensorMap(keys_first_block, [tensor[0].copy()])

    @pytest.fixture
    def tensor_single_block_extra_grad(self, tensor):
        block_extra_grad = tensor[0].copy()
        gradient = tensor.block(0).gradient("positions")
        block_extra_grad.add_gradient(
            "foo", gradient.data, gradient.samples, gradient.components
        )

        keys_first_block = Labels(
            names=tensor.keys.names,
            values=np.array(tensor.keys[0].tolist()).reshape(1, -1),
        )

        return TensorMap(keys_first_block, [block_extra_grad])

    def test_wrong_type(self, tensor):
        """Test if a wrong type (e.g., TensorMap) is provided."""
        with pytest.raises(TypeError, match="list or a tuple"):
            equistore.join(tensor, axis="properties")

    def test_single_tensormap(self, tensor):
        """Test if only one TensorMap is provided."""
        tensor_joined = equistore.join([tensor], axis="properties")
        assert tensor_joined is tensor

    @pytest.mark.parametrize("tensor", ([], ()))
    def test_no_tensormaps(self, tensor):
        """Test if an empty list or tuple is provided."""
        with pytest.raises(ValueError, match="provide at least one"):
            equistore.join(tensor, axis="properties")

    def test_join_properties(self, tensor):
        """Test public join function with three tensormaps along `properties`.

        We check for the values below."""

        tensor_joined = equistore.join([tensor, tensor, tensor], axis="properties")

        # test property names
        names = tensor.block(0).properties.names
        assert tensor_joined.block(0).properties.names == ("tensor",) + names

        # test property values
        tensor_prop = np.unique(tensor_joined.block(0).properties["tensor"])
        assert set(tensor_prop) == set((0, 1, 2))

    def test_join_properties_with_different_props(self, tensor, components_tensor):
            """Test public join function with three tensormaps along `properties`.

            We check for the values below."""

            tensor_joined = equistore.join([tensor, tensor, tensor], axis="properties")

            # test property names
            names = tensor.block(0).properties.names
            assert tensor_joined.block(0).properties.names == ("tensor",) + names

            # test property values
            tensor_prop = np.unique(tensor_joined.block(0).properties["tensor"])
            assert set(tensor_prop) == set((0, 1, 2))

    def test_join_samples(self, tensor):
        """Test public join function with three tensormaps along `samples`."""
        tensor_joined = equistore.join([tensor, tensor, tensor], axis="samples")

        # test sample values
        assert len(tensor_joined.block(0).samples) == 3 * len(tensor.block(0).samples)

    def test_join_error(self, tensor):
        """Test error with unknown `axis` keyword."""
        with pytest.raises(ValueError, match="values for the `axis` parameter"):
            equistore.join([tensor, tensor, tensor], axis="foo")

    def test_join_properties_values(self, tensor):
        """Test values for joining along `properties`."""
        ts_1 = equistore.slice(tensor, properties=tensor[0].properties[:1])
        ts_2 = equistore.slice(tensor, properties=tensor[0].properties[1:])

        tensor_joined = equistore.join([ts_1, ts_2], axis="properties")
        for i, block_tensor in tensor:
            block_tensor_joined = tensor_joined[i]

            assert_equal(block_tensor_joined.values, block_tensor.values)

    def test_join_properties_different_gradients(
        self, tensor_single_block, tensor_single_block_extra_grad
    ):
        """Test error raise if `gradients` are not the same."""
        with pytest.raises(EquistoreError, match="gradient"):
            equistore.join(
                [tensor_single_block, tensor_single_block_extra_grad], axis="properties"
            )

    def test_join_samples_values(self, tensor):
        """Test values for joining along `samples`."""
        keys = Labels(
            names=tensor.keys.names,
            values=np.array(tensor.keys[0].tolist()).reshape(1, -1),
        )

        tm = TensorMap(keys, [tensor[0].copy()])
        ts_1 = equistore.slice(tm, samples=tensor[0].samples[:1])
        ts_2 = equistore.slice(tm, samples=tensor[0].samples[1:])

        tensor_joined = equistore.join([ts_1, ts_2], axis="samples")
        for i, block_tensor in tm:
            block_tensor_joined = tensor_joined[i]

            assert_equal(block_tensor_joined.values, block_tensor.values)

    @pytest.mark.parametrize("axis", ["samples", "properties"])
    def test_join_samples_different_components(self, components_tensor, axis):
        """Test error raise if `components` are not the same."""
        components_tensor_c2p = components_tensor.copy().components_to_properties(
            ["spherical_harmonics_m"]
        )

        with pytest.raises(EquistoreError, match="components"):
            equistore.join([components_tensor_c2p, components_tensor], axis=axis)

    @pytest.mark.parametrize("axis", ["samples", "properties"])
    def test_join_samples_different_gradients(
        self,
        tensor_single_block,
        tensor_single_block_extra_grad,
        axis,
    ):
        """Test error raise if `gradients` are not the same."""
        with pytest.raises(EquistoreError, match="gradient"):
            equistore.join(
                [tensor_single_block, tensor_single_block_extra_grad], axis=axis
            )


class TestJoinLabels:
    """Test edge cases of label joining."""

    @pytest.fixture
    def sample_labels(self):
        return Labels(names=["prop"], values=np.arange(2).reshape(-1, 1))

    @pytest.fixture
    def keys(self):
        return Labels(names=["prop"], values=np.arange(1).reshape(-1, 1))

    def test_same_names_same_values(self, sample_labels, keys):
        """Test Label joining using labels with same names but same values."""

        names = ("structure", "prop_1")
        property_labels = Labels(names, np.vstack([np.arange(5), np.arange(5)]).T)

        block = TensorBlock(
            values=np.zeros([2, 5]),
            samples=sample_labels,
            components=[],
            properties=property_labels,
        )

        tm = TensorMap(keys, [block])
        joined_tm = equistore.join([tm, tm], axis="properties")

        joined_labels = joined_tm.block(0).properties

        assert joined_labels.names == ("tensor",) + names

        ref = np.array(
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 2, 2],
                [0, 3, 3],
                [0, 4, 4],
                [1, 0, 0],
                [1, 1, 1],
                [1, 2, 2],
                [1, 3, 3],
                [1, 4, 4],
            ]
        )

        assert_equal(joined_labels.tolist(), ref)

    def test_same_names_unique_values(self, sample_labels, keys):
        """Test Label joining using labels with same names and unique values."""
        names = ("structure", "prop_1")
        property_labels_1 = Labels(names, np.vstack([np.arange(5), np.arange(5)]).T)
        property_labels_2 = Labels(
            names, np.vstack([np.arange(5, 10), np.arange(5, 10)]).T
        )

        block_1 = TensorBlock(
            values=np.zeros([2, 5]),
            samples=sample_labels,
            components=[],
            properties=property_labels_1,
        )

        block_2 = TensorBlock(
            values=np.zeros([2, 5]),
            samples=sample_labels,
            components=[],
            properties=property_labels_2,
        )

        joined_tm = equistore.join(
            [TensorMap(keys, [block_1]), TensorMap(keys, [block_2])],
            axis="properties",
        )

        joined_labels = joined_tm.block(0).properties

        assert joined_labels.names == ("structure", "prop_1")
        assert_equal(
            joined_labels.tolist(), np.vstack([np.arange(10), np.arange(10)]).T
        )

    def test_different_names(self, sample_labels, keys):
        """Test Label joining using labels with different names."""
        values = np.vstack([np.arange(5), np.arange(5)]).T
        property_labels_1 = Labels(("structure", "prop_1"), -values)
        property_labels_2 = Labels(("structure", "prop_2"), values)

        block_1 = TensorBlock(
            values=np.zeros([2, 5]),
            samples=sample_labels,
            components=[],
            properties=property_labels_1,
        )

        block_2 = TensorBlock(
            values=np.zeros([2, 5]),
            samples=sample_labels,
            components=[],
            properties=property_labels_2,
        )

        joined_tm = equistore.join(
            [TensorMap(keys, [block_1]), TensorMap(keys, [block_2])],
            axis="properties",
        )

        joined_labels = joined_tm.block(0).properties

        assert joined_labels.names == ("tensor", "property")

        ref = np.array(
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
            ]
        )

        assert_equal(joined_labels.tolist(), ref)

    def test_different_names_different_length(self, sample_labels, keys):
        """Test Label joining using labels with different names and different length."""
        property_labels_1 = Labels(
            ("structure", "prop_1"), np.vstack(2 * [np.arange(5)]).T
        )
        property_labels_2 = Labels(
            ("structure", "prop_2", "prop_3"), np.vstack(3 * [np.arange(5)]).T
        )

        block_1 = TensorBlock(
            values=np.zeros([2, 5]),
            samples=sample_labels,
            components=[],
            properties=property_labels_1,
        )

        block_2 = TensorBlock(
            values=np.zeros([2, 5]),
            samples=sample_labels,
            components=[],
            properties=property_labels_2,
        )

        joined_tm = equistore.join(
            [TensorMap(keys, [block_1]), TensorMap(keys, [block_2])],
            axis="properties",
        )

        joined_labels = joined_tm.block(0).properties

        assert joined_labels.names == ("tensor", "property")

        ref = np.array(
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
            ]
        )

        assert_equal(joined_labels.tolist(), ref)
