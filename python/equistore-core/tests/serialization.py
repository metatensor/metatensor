import os
import pickle
import sys

import numpy as np
import pytest
from numpy.testing import assert_equal

import equistore.core
from equistore.core import Labels, TensorBlock, TensorMap

from . import utils


@pytest.fixture
def tensor():
    return utils.tensor()


@pytest.mark.parametrize("use_numpy", (True, False))
def test_load(use_numpy):
    tensor = equistore.core.load(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "equistore",
            "tests",
            "data.npz",
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
    assert gradient.values.shape == (59, 3, 5, 3)


# using tmpdir as pytest-built-in fixture
# https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html#the-tmpdir-and-tmpdir-factory-fixtures
@pytest.mark.parametrize("use_numpy", (True, False))
def test_save(use_numpy, tmpdir, tensor):
    """Check that as saved file loads fine with numpy."""
    tmpfile = "serialize-test.npz"

    with tmpdir.as_cwd():
        equistore.core.save(tmpfile, tensor, use_numpy=use_numpy)
        data = np.load(tmpfile)

    assert len(data.keys()) == 29

    assert_equal(data["keys"], tensor.keys)
    for i, (_, block) in enumerate(tensor):
        prefix = f"blocks/{i}"
        assert_equal(data[f"{prefix}/values"], block.values)
        assert_equal(data[f"{prefix}/samples"], block.samples)
        assert_equal(data[f"{prefix}/components/0"], block.components[0])
        assert_equal(data[f"{prefix}/properties"], block.properties)

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            prefix = f"blocks/{i}/gradients/{parameter}"
            assert_equal(data[f"{prefix}/values"], gradient.values)
            assert_equal(data[f"{prefix}/samples"], gradient.samples)
            assert_equal(data[f"{prefix}/components/0"], gradient.components[0])


def test_save_warning_errors(tmpdir, tensor):
    # does not have .npz ending and causes warning
    tmpfile = "serialize-test"

    with pytest.warns() as record:
        with tmpdir.as_cwd():
            equistore.core.save(tmpfile, tensor)

    expected = f"adding '.npz' extension, the file will be saved at '{tmpfile}.npz'"
    assert str(record[0].message) == expected

    tmpfile = "serialize-test.npz"

    message = (
        "tensor should be a 'TensorMap', not <class 'equistore.core.block.TensorBlock'>"
    )
    with pytest.raises(TypeError, match=message):
        with tmpdir.as_cwd():
            equistore.core.save(tmpfile, tensor.block(0))


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
    for key, block in tensor:
        ref_block = tensor.block(key)
        assert_equal(type(block.values), type(ref_block.values))
        assert_equal(block.values, ref_block.values)
        assert_equal(block.samples, ref_block.samples)
        assert_equal(block.components, ref_block.components)
        assert_equal(block.properties, ref_block.properties)

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            ref_gradient = ref_block.gradient(parameter)
            assert_equal(gradient.values, ref_gradient.values)
            assert_equal(gradient.samples, ref_gradient.samples)
            assert_equal(gradient.components, ref_gradient.components)


@pytest.mark.parametrize("use_numpy", (True, False))
def test_nested_gradients(tmpdir, use_numpy):
    block = TensorBlock(
        values=np.random.rand(3, 3),
        samples=Labels.arange("s", 3),
        components=[],
        properties=Labels.arange("p", 3),
    )

    grad = TensorBlock(
        values=np.random.rand(3, 3),
        samples=Labels.arange("sample", 3),
        components=[],
        properties=Labels.arange("p", 3),
    )

    grad_grad = TensorBlock(
        values=np.random.rand(3, 5, 3),
        samples=Labels.arange("sample", 3),
        components=[Labels.arange("c", 5)],
        properties=Labels.arange("p", 3),
    )

    grad.add_gradient("grad-of-grad", grad_grad)
    block.add_gradient("grad", grad)
    tensor = TensorMap(Labels.single(), [block])

    tmpfile = "grad-grad-test.npz"

    with tmpdir.as_cwd():
        equistore.core.save(tmpfile, tensor, use_numpy=use_numpy)

        # load back with numpy
        data = np.load(tmpfile)

        # load back with equistore
        loaded = equistore.core.load(tmpfile)

    assert_equal(data["keys"], tensor.keys)
    assert_equal(data["keys"], loaded.keys)

    for i, (key, block) in enumerate(tensor):
        loaded_block = loaded.block(key)

        prefix = f"blocks/{i}"
        assert_equal(data[f"{prefix}/values"], block.values)
        assert_equal(data[f"{prefix}/values"], loaded_block.values)

        assert_equal(data[f"{prefix}/samples"], block.samples)
        assert_equal(data[f"{prefix}/samples"], loaded_block.samples)

        assert_equal(data[f"{prefix}/properties"], block.properties)
        assert_equal(data[f"{prefix}/properties"], loaded_block.properties)

        assert_equal(block.gradients_list(), loaded_block.gradients_list())

        for parameter, gradient in block.gradients():
            loaded_gradient = loaded_block.gradient(parameter)
            grad_prefix = f"{prefix}/gradients/{parameter}"

            assert_equal(data[f"{grad_prefix}/values"], gradient.values)
            assert_equal(data[f"{grad_prefix}/values"], loaded_gradient.values)

            assert_equal(data[f"{grad_prefix}/samples"], gradient.samples)
            assert_equal(data[f"{grad_prefix}/samples"], loaded_gradient.samples)

            assert len(gradient.components) == len(loaded_gradient.components)
            for a, b in zip(gradient.components, loaded_gradient.components):
                assert_equal(a, b)

            assert_equal(gradient.properties, loaded_gradient.properties)

            assert_equal(gradient.gradients_list(), loaded_gradient.gradients_list())

            for parameter, grad_grad in gradient.gradients():
                loaded_grad_grad = loaded_gradient.gradient(parameter)
                grad_grad_prefix = f"{grad_prefix}/gradients/{parameter}"

                assert_equal(data[f"{grad_grad_prefix}/values"], grad_grad.values)
                assert_equal(
                    data[f"{grad_grad_prefix}/values"], loaded_grad_grad.values
                )

                assert_equal(data[f"{grad_grad_prefix}/samples"], grad_grad.samples)
                assert_equal(
                    data[f"{grad_grad_prefix}/samples"], loaded_grad_grad.samples
                )

                assert len(grad_grad.components) == len(loaded_grad_grad.components)
                for a, b in zip(grad_grad.components, loaded_grad_grad.components):
                    assert_equal(a, b)

                assert_equal(grad_grad.properties, loaded_grad_grad.properties)
