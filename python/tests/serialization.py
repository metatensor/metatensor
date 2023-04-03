import os

import numpy as np
import pytest
from numpy.testing import assert_equal

import equistore
from equistore import TensorMap


ROOT = os.path.dirname(__file__)


class TestIo:
    @pytest.mark.parametrize("use_numpy", (True, False))
    def test_load(self, use_numpy):
        tensor = equistore.load(
            os.path.join(ROOT, "..", "..", "equistore-core", "tests", "data.npz"),
            use_numpy=use_numpy,
        )

        assert isinstance(tensor, TensorMap)
        assert tensor.keys.names == (
            "spherical_harmonics_l",
            "center_species",
            "neighbor_species",
        )
        assert len(tensor.keys) == 27

        block = tensor.block(
            spherical_harmonics_l=2, center_species=6, neighbor_species=1
        )
        assert block.samples.names == ("structure", "center")
        assert block.values.shape == (9, 5, 3)

        gradient = block.gradient("positions")
        assert gradient.samples.names == ("sample", "structure", "atom")
        assert gradient.data.shape == (59, 3, 5, 3)

    @pytest.mark.parametrize("use_numpy", (True, False))
    def test_save(self, use_numpy, tmpdir, tensor):
        """Check that as saved file loads fine with numpy."""
        tmpfile = "serialize-test.npz"

        with tmpdir.as_cwd():
            equistore.save(tmpfile, tensor, use_numpy=use_numpy)
            data = np.load(tmpfile)

        assert len(data.keys()) == 29

        assert_equal(data["keys"], tensor.keys)
        for i, (_, block) in enumerate(tensor):
            prefix = f"blocks/{i}/values"
            assert_equal(data[f"{prefix}/data"], block.values)
            assert_equal(data[f"{prefix}/samples"], block.samples)
            assert_equal(data[f"{prefix}/components/0"], block.components[0])
            assert_equal(data[f"{prefix}/properties"], block.properties)

            for parameter in block.gradients_list():
                gradient = block.gradient(parameter)
                prefix = f"blocks/{i}/gradients/{parameter}"
                assert_equal(data[f"{prefix}/data"], gradient.data)
                assert_equal(data[f"{prefix}/samples"], gradient.samples)
                assert_equal(data[f"{prefix}/components/0"], gradient.components[0])
