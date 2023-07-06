import io
import os
import pickle
import sys

import numpy as np
import pytest

import equistore.core
from equistore.core import Labels, TensorBlock, TensorMap

from . import utils


@pytest.fixture
def tensor():
    return utils.tensor()


@pytest.mark.parametrize("use_numpy", (True, False))
@pytest.mark.parametrize("memory_buffer", (True, False))
def test_load(use_numpy, memory_buffer):
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "equistore",
        "tests",
        "data.npz",
    )

    if memory_buffer:
        with open(path, "rb") as fd:
            buffer = fd.read()

        assert isinstance(buffer, bytes)
        file = io.BytesIO(buffer)
    else:
        file = path

    tensor = equistore.core.load(
        file,
        use_numpy=use_numpy,
    )

    assert isinstance(tensor, TensorMap)
    assert tensor.keys.names == [
        "spherical_harmonics_l",
        "center_species",
        "neighbor_species",
    ]
    assert len(tensor.keys) == 27

    block = tensor.block(spherical_harmonics_l=2, center_species=6, neighbor_species=1)
    assert block.samples.names == ["structure", "center"]
    assert block.values.shape == (9, 5, 3)

    gradient = block.gradient("positions")
    assert gradient.samples.names == ["sample", "structure", "atom"]
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

    assert _npz_labels(data["keys"]) == tensor.keys
    for i, block in enumerate(tensor.blocks()):
        prefix = f"blocks/{i}"

        np.testing.assert_equal(data[f"{prefix}/values"], block.values)
        assert _npz_labels(data[f"{prefix}/samples"]) == block.samples
        assert _npz_labels(data[f"{prefix}/components/0"]) == block.components[0]
        assert _npz_labels(data[f"{prefix}/properties"]) == block.properties

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            prefix = f"blocks/{i}/gradients/{parameter}"

            np.testing.assert_equal(data[f"{prefix}/values"], gradient.values)
            assert _npz_labels(data[f"{prefix}/samples"]) == gradient.samples
            assert _npz_labels(data[f"{prefix}/components/0"]) == gradient.components[0]


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
    Checks that pickling and unpickling a tensor map results in the same tensor map
    """

    tmpfile = "serialize-test.pickle"

    with tmpdir.as_cwd():
        with open(tmpfile, "wb") as f:
            pickle.dump(tensor, f, protocol=protocol)

        with open(tmpfile, "rb") as f:
            tensor_loaded = pickle.load(f)

    np.testing.assert_equal(tensor.keys, tensor_loaded.keys)
    for key, block in tensor_loaded.items():
        ref_block = tensor.block(key)

        assert isinstance(block.values, np.ndarray)
        np.testing.assert_equal(block.values, ref_block.values)

        assert block.samples == ref_block.samples
        assert block.components == ref_block.components
        assert block.properties == ref_block.properties

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            ref_gradient = ref_block.gradient(parameter)

            np.testing.assert_equal(gradient.values, ref_gradient.values)

            assert gradient.samples == ref_gradient.samples
            assert gradient.components == ref_gradient.components


@pytest.mark.parametrize("use_numpy", (True, False))
def test_nested_gradients(tmpdir, use_numpy):
    block = TensorBlock(
        values=np.random.rand(3, 3),
        samples=Labels.range("s", 3),
        components=[],
        properties=Labels.range("p", 3),
    )

    grad = TensorBlock(
        values=np.random.rand(3, 3),
        samples=Labels.range("sample", 3),
        components=[],
        properties=Labels.range("p", 3),
    )

    grad_grad = TensorBlock(
        values=np.random.rand(3, 5, 3),
        samples=Labels.range("sample", 3),
        components=[Labels.range("c", 5)],
        properties=Labels.range("p", 3),
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

    assert _npz_labels(data["keys"]) == tensor.keys
    assert _npz_labels(data["keys"]) == loaded.keys

    for i, (key, block) in enumerate(tensor.items()):
        loaded_block = loaded.block(key)

        prefix = f"blocks/{i}"
        np.testing.assert_equal(data[f"{prefix}/values"], block.values)
        np.testing.assert_equal(block.values, loaded_block.values)

        assert _npz_labels(data[f"{prefix}/samples"]) == block.samples
        assert block.samples == loaded_block.samples

        assert _npz_labels(data[f"{prefix}/properties"]) == block.properties
        assert block.properties == loaded_block.properties

        assert block.gradients_list() == loaded_block.gradients_list()

        for parameter, gradient in block.gradients():
            loaded_gradient = loaded_block.gradient(parameter)
            grad_prefix = f"{prefix}/gradients/{parameter}"

            np.testing.assert_equal(data[f"{grad_prefix}/values"], gradient.values)
            np.testing.assert_equal(gradient.values, loaded_gradient.values)

            assert _npz_labels(data[f"{grad_prefix}/samples"]) == gradient.samples
            assert gradient.samples == loaded_gradient.samples

            assert gradient.components == loaded_gradient.components
            assert gradient.properties == loaded_gradient.properties

            assert gradient.gradients_list() == loaded_gradient.gradients_list()

            for parameter, grad_grad in gradient.gradients():
                loaded_grad_grad = loaded_gradient.gradient(parameter)
                grad_grad_prefix = f"{grad_prefix}/gradients/{parameter}"

                np.testing.assert_equal(
                    data[f"{grad_grad_prefix}/values"], grad_grad.values
                )
                np.testing.assert_equal(grad_grad.values, loaded_grad_grad.values)

                assert (
                    _npz_labels(data[f"{grad_grad_prefix}/samples"])
                    == grad_grad.samples
                )

                assert grad_grad.samples == loaded_grad_grad.samples

                assert len(grad_grad.components) == len(loaded_grad_grad.components)
                for a, b in zip(grad_grad.components, loaded_grad_grad.components):
                    assert a == b

                assert grad_grad.properties == loaded_grad_grad.properties


def _npz_labels(data):
    names = data.dtype.names
    return Labels(names=names, values=data.view(dtype=np.int32).reshape(-1, len(names)))
