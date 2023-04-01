from os import path

import numpy as np
import pytest

import equistore


DATA_ROOT = path.join(path.dirname(__file__), "..", "data")


class TestBlockFromArray:
    @pytest.mark.parametrize("n_axes", [0, 1])
    def test_too_few_axes(self, n_axes):
        """Test block_from_array when too few axes are provided."""
        with pytest.raises(ValueError, match="at least"):
            equistore.block_from_array(np.zeros((4,) * n_axes))

    def test_without_components(self):
        """Test block_from_array for a 2D array."""
        array = np.zeros((6, 7))
        tblock = equistore.block_from_array(array)
        assert tblock.values is array

        assert tblock.samples.names == ("sample",)
        np.testing.assert_equal(
            tblock.samples.asarray(), np.arange(array.shape[0]).reshape((-1, 1))
        )

        assert tblock.properties.names == ("property",)
        np.testing.assert_equal(
            tblock.properties.asarray(), np.arange(array.shape[1]).reshape((-1, 1))
        )

    def test_with_components(self):
        """Test block_from_array with components."""
        array = array = np.zeros((6, 5, 7))
        tblock = equistore.block_from_array(array)
        assert tblock.values is array

        assert tblock.samples.names == ("sample",)
        np.testing.assert_equal(
            tblock.samples.asarray(), np.arange(array.shape[0]).reshape((-1, 1))
        )

        assert len(tblock.components) == 1
        component = tblock.components[0]
        assert component.names == ("component_1",)
        np.testing.assert_equal(
            component.asarray(), np.arange(array.shape[1]).reshape((-1, 1))
        )

        assert tblock.properties.names == ("property",)
        np.testing.assert_equal(
            tblock.properties.asarray(), np.arange(array.shape[2]).reshape((-1, 1))
        )
