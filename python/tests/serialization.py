import os
import pickle
import sys

import numpy as np
import pytest
from numpy.testing import assert_equal

import equistore
from equistore import TensorMap

from . import utils


@pytest.fixture
def tensor():
    return utils.tensor()


@pytest.mark.parametrize("use_numpy", (True, False))
def test_load(use_numpy):
    tensor = equistore.load(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "equistore-core", "tests", "data.npz"
        ),
        use_numpy=use_numpy,
    )

    assert isinstance(tensor, TensorMap)
    assert tensor.keys.names == (
        "spherical_harmonics_l",
        "center_species",
        "neighbor_species",
    )
    assert len(tensor.keys) == 27

    block = tensor.block(spherical_harmonics_l=2, center_species=6, neighbor_species=1)
    assert block.samples.names == ("structure", "center")
    assert block.values.shape == (9, 5, 3)

    gradient = block.gradient("positions")
    assert gradient.samples.names == ("sample", "structure", "atom")
    assert gradient.data.shape == (59, 3, 5, 3)


@pytest.mark.parametrize("use_numpy", (True, False))
def test_save(use_numpy, tmpdir, tensor):
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


if (sys.version_info.major >= 3) and (sys.version_info.minor >= 8):
    protocols = (4, 5)
else:
    protocols = (4,)


@pytest.mark.parametrize("protocol", protocols)
def test_pickle(protocol, tmpdir, tensor):
    """
    Checks that pickling and unpickling a tensor map
    results in the same tensor map
    """

    tmpfile = "serialize-test.pickle"

    with tmpdir.as_cwd():
        with open(tmpfile, "wb") as f:
            pickle.dump(tensor, f, protocol=protocol)

        with open(tmpfile, "rb") as f:
            tensor_loaded = pickle.load(f)

    assert_equal(tensor.keys, tensor_loaded.keys)
    assert_equal(len(tensor.blocks()), len(tensor_loaded.blocks()))
    for i, (_, block) in enumerate(tensor):
        ref_block = tensor.blocks()[i]
        assert_equal(type(block.values), type(ref_block.values))
        assert_equal(block.values, ref_block.values)
        assert_equal(block.samples, ref_block.samples)
        assert_equal(block.components, ref_block.components)
        assert_equal(block.properties, ref_block.properties)

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            ref_gradient = ref_block.gradient(parameter)
            assert_equal(gradient.data, ref_gradient.data)
            assert_equal(gradient.samples, ref_gradient.samples)
            assert_equal(gradient.components, ref_gradient.components)
