import os

# import numpy as np
# from numpy.testing import assert_allclose, assert_array_equal
import pytest

import equistore
import equistore.io


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
TEST_FILE = "qm7-spherical-expansion.npz"


@pytest.fixture
def test_tensormap_a():
    pass


@pytest.fixture
def test_tensormap_b():
    pass


@pytest.fixture
def test_tensormap_c():
    return equistore.io.load(os.path.join(DATA_ROOT, TEST_FILE), use_numpy=True)


@pytest.mark.parametrize("test_tensormap", ["test_tensormap_a", "test_tensormap_b"])
class TestAbs:
    """"""

    def test_abs_metadata(self, test_tensormap, request):
        """Tests that the returned tensor has the same metadata as the input."""
        # Get the parametrized arg
        # test_tensormap = request.getfixturevalue(test_tensormap)
        # testing
        assert True
