import os
from typing import Union

import numpy as np
import pytest
import torch
from packaging import version

import metatensor.torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from . import _tests_utils


@pytest.fixture
def tensor_path():
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "data.npz",
    )


@pytest.fixture
def block_path():
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "block.npz",
    )


@pytest.fixture
def labels_path():
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "keys.npy",
    )


def check_labels(labels):
    assert labels.names == [
        "o3_lambda",
        "o3_sigma",
        "center_type",
        "neighbor_type",
    ]
    assert len(labels) == 27


def check_block(block):
    assert block.samples.names == ["system", "atom"]
    assert block.values.shape == (9, 5, 3)

    gradient = block.gradient("positions")
    assert gradient.samples.names == ["sample", "system", "atom"]
    assert gradient.values.shape == (59, 3, 5, 3)


def check_tensor(tensor):
    check_labels(tensor.keys)

    block = tensor.block(dict(o3_lambda=2, center_type=6, neighbor_type=1))
    check_block(block)


def test_load(tensor_path):
    loaded = metatensor.torch.load(tensor_path)
    check_tensor(loaded)

    loaded = TensorMap.load(tensor_path)
    check_tensor(loaded)

    # load from buffer
    buffer = torch.tensor(np.fromfile(tensor_path, dtype="uint8"))

    loaded = metatensor.torch.load_buffer(buffer)
    check_tensor(loaded)

    loaded = TensorMap.load_buffer(buffer)
    check_tensor(loaded)


def test_save(tmpdir, tensor_path):
    """Check that we can save and load a tensor to a file"""
    tmpfile = "serialize-test.npz"

    tensor = _tests_utils.tensor(dtype=torch.float64)

    with tmpdir.as_cwd():
        metatensor.torch.save(tmpfile, tensor)
        data = metatensor.torch.load(tmpfile)

        assert len(data.keys) == 4

        tensor.save(tmpfile)
        data = metatensor.torch.load(tmpfile)

        assert len(data.keys) == 4

    # save using buffer
    buffer = torch.tensor(np.fromfile(tensor_path, dtype="uint8"))
    tensor = metatensor.torch.load_buffer(buffer)

    saved = metatensor.torch.save_buffer(tensor)
    assert torch.all(buffer == saved)

    saved = tensor.save_buffer()
    assert torch.all(buffer == saved)


def test_pickle(tmpdir, tensor_path):
    tensor = metatensor.torch.load(tensor_path)
    tmpfile = "serialize-test.npz"

    with tmpdir.as_cwd():
        torch.save(tensor, tmpfile)
        if version.parse(torch.__version__) <= version.parse("1.12"):
            loaded = torch.load(tmpfile, weights_only=False)
        else:
            loaded = torch.load(tmpfile)

    check_tensor(loaded)


def test_save_load_zero_length_block(tmpdir):
    """
    Tests that attempting to save and load a TensorMap with a zero-length axis block
    does not raise an error.
    """
    tensor_zero_len_block = _tests_utils.tensor_zero_len_block()

    with tmpdir.as_cwd():
        file = "serialize-test-zero-len-block.npz"
        metatensor.torch.save(file, tensor_zero_len_block)
        metatensor.torch.load(file)


def test_load_block(block_path):
    loaded = metatensor.torch.load_block(block_path)
    check_block(loaded)

    loaded = TensorBlock.load(block_path)
    check_block(loaded)

    # load from buffer
    buffer = torch.tensor(np.fromfile(block_path, dtype="uint8"))

    loaded = metatensor.torch.load_block_buffer(buffer)
    check_block(loaded)

    loaded = TensorBlock.load_buffer(buffer)
    check_block(loaded)


def test_save_block(tmpdir, block_path):
    """Check that we can save and load a tensor to a file"""
    tmpfile = "serialize-test.npz"

    block = _tests_utils.tensor(dtype=torch.float64).block(2)

    with tmpdir.as_cwd():
        metatensor.torch.save(tmpfile, block)
        data = metatensor.torch.load_block(tmpfile)
        assert len(data) == 4

        block.save(tmpfile)
        data = metatensor.torch.load_block(tmpfile)
        assert len(data) == 4

    # save with buffer
    buffer = torch.tensor(np.fromfile(block_path, dtype="uint8"))
    tensor = metatensor.torch.load_block_buffer(buffer)

    saved = metatensor.torch.save_buffer(tensor)
    assert torch.all(buffer == saved)

    saved = tensor.save_buffer()
    assert torch.all(buffer == saved)


def test_pickle_block(tmpdir, block_path):
    block = metatensor.torch.load_block(block_path)
    tmpfile = "serialize-test.npz"

    with tmpdir.as_cwd():
        torch.save(block, tmpfile)

        if version.parse(torch.__version__) <= version.parse("1.12"):
            loaded = torch.load(tmpfile, weights_only=False)
        else:
            loaded = torch.load(tmpfile)

    check_block(loaded)


def test_load_labels(labels_path):
    loaded = metatensor.torch.load_labels(labels_path)
    check_labels(loaded)

    loaded = metatensor.torch.Labels.load(labels_path)
    check_labels(loaded)

    # load from buffer
    buffer = torch.tensor(np.fromfile(labels_path, dtype="uint8"))

    loaded = metatensor.torch.load_labels_buffer(buffer)
    check_labels(loaded)

    loaded = Labels.load_buffer(buffer)
    check_labels(loaded)


def test_save_labels(tmpdir, labels_path):
    """Check that we can save and load a tensor to a file"""
    tmpfile = "serialize-test.npy"

    labels = _tests_utils.tensor(dtype=torch.float64).keys

    with tmpdir.as_cwd():
        metatensor.torch.save(tmpfile, labels)
        data = metatensor.torch.load_labels(tmpfile)
        assert len(data) == 4

        labels.save(tmpfile)
        data = metatensor.torch.load_labels(tmpfile)
        assert len(data) == 4

    # save with buffer
    buffer = torch.tensor(np.fromfile(labels_path, dtype="uint8"))
    tensor = metatensor.torch.load_labels_buffer(buffer)

    saved = metatensor.torch.save_buffer(tensor)
    assert torch.all(buffer == saved)

    saved = tensor.save_buffer()
    assert torch.all(buffer == saved)


def test_pickle_labels(tmpdir, labels_path):
    labels = metatensor.torch.load_labels(labels_path)
    tmpfile = "serialize-test.npy"

    with tmpdir.as_cwd():
        torch.save(labels, tmpfile)

        if version.parse(torch.__version__) <= version.parse("1.12"):
            loaded = torch.load(tmpfile, weights_only=False)
        else:
            loaded = torch.load(tmpfile)

    check_labels(loaded)


class Serialization:
    def load(self, path: str) -> TensorMap:
        return metatensor.torch.load(path=path)

    def load_buffer(self, buffer: torch.Tensor) -> TensorMap:
        return metatensor.torch.load_buffer(buffer=buffer)

    def load_block(self, path: str) -> TensorBlock:
        return metatensor.torch.load_block(path=path)

    def load_block_buffer(self, buffer: torch.Tensor) -> TensorBlock:
        return metatensor.torch.load_block_buffer(buffer=buffer)

    def load_labels(self, path: str) -> Labels:
        return metatensor.torch.load_labels(path=path)

    def load_labels_buffer(self, buffer: torch.Tensor) -> Labels:
        return metatensor.torch.load_labels_buffer(buffer=buffer)

    def save(self, path: str, data: Union[Labels, TensorBlock, TensorMap]):
        return metatensor.torch.save(path=path, data=data)

    def save_buffer(self, data: Union[Labels, TensorBlock, TensorMap]) -> torch.Tensor:
        return metatensor.torch.save_buffer(data=data)


def test_script():
    # check that the operators definition (in register.cpp) match what we expect
    # (defined in `Serialization`)
    _ = torch.jit.script(Serialization)
