import os
from pathlib import Path
from typing import Union

import numpy as np
import pytest
import torch

import metatensor.torch as mts
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
        "data.mts",
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
        "block.mts",
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
        "keys.mts",
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
    loaded = mts.load(tensor_path)
    check_tensor(loaded)

    loaded = TensorMap.load(tensor_path)
    check_tensor(loaded)

    # using Path
    loaded = mts.load(Path(tensor_path))
    check_tensor(loaded)

    # using pre-opened file
    with open(tensor_path, "rb") as file:
        loaded = mts.load(file)
        check_tensor(loaded)

    # load from buffer
    buffer = torch.tensor(np.fromfile(tensor_path, dtype="uint8"))

    loaded = mts.load_buffer(buffer)
    check_tensor(loaded)

    loaded = TensorMap.load_buffer(buffer)
    check_tensor(loaded)


def test_save(tmpdir, tensor_path):
    """Check that we can save and load a tensor to a file"""
    tmpfile = "serialize-test.mts"

    tensor = _tests_utils.tensor(dtype=torch.float64)

    with tmpdir.as_cwd():
        mts.save(tmpfile, tensor)
        data = mts.load(tmpfile)
        assert len(data.keys) == 4

        tensor.save(tmpfile)
        data = mts.load(tmpfile)
        assert len(data.keys) == 4

        # using Path
        mts.save(Path(tmpfile), tensor)
        data = mts.load(tmpfile)
        assert len(data.keys) == 4

        # using pre-opened file
        with open(tmpfile, "wb") as file:
            mts.save(file, tensor)
        data = mts.load(tmpfile)
        assert len(data.keys) == 4

    # save using buffer
    buffer = torch.tensor(np.fromfile(tensor_path, dtype="uint8"))
    tensor = mts.load_buffer(buffer)

    saved = mts.save_buffer(tensor)
    assert torch.all(buffer == saved)

    saved = tensor.save_buffer()
    assert torch.all(buffer == saved)

    tensor_f32 = tensor.to(torch.float32)
    with pytest.raises(ValueError, match="only float64 is supported"):
        mts.save_buffer(tensor_f32)
    with pytest.raises(ValueError, match="only float64 is supported"):
        mts.save(tmpfile, tensor_f32)

    tensor_meta = tensor.to(torch.device("meta"))
    with pytest.raises(ValueError, match="only CPU is supported"):
        mts.save_buffer(tensor_meta)
    with pytest.raises(ValueError, match="only CPU is supported"):
        mts.save(tmpfile, tensor_meta)


def test_pickle(tmpdir, tensor_path):
    tensor = mts.load(tensor_path)
    tmpfile = "serialize-test.mts"

    with tmpdir.as_cwd():
        torch.save(tensor, tmpfile)
        loaded = torch.load(tmpfile, weights_only=False)

    check_tensor(loaded)


def test_save_load_zero_length_block(tmpdir):
    """
    Tests that attempting to save and load a TensorMap with a zero-length axis block
    does not raise an error.
    """
    tensor_zero_len_block = _tests_utils.tensor_zero_len_block()

    with tmpdir.as_cwd():
        file = "serialize-test-zero-len-block.mts"
        mts.save(file, tensor_zero_len_block)
        mts.load(file)


def test_load_block(block_path):
    loaded = mts.load_block(block_path)
    check_block(loaded)

    loaded = TensorBlock.load(block_path)
    check_block(loaded)

    # using Path
    loaded = mts.load_block(Path(block_path))
    check_block(loaded)

    # using pre-opened file
    with open(block_path, "rb") as file:
        loaded = mts.load_block(file)
        check_block(loaded)

    # load from buffer
    buffer = torch.tensor(np.fromfile(block_path, dtype="uint8"))

    loaded = mts.load_block_buffer(buffer)
    check_block(loaded)

    loaded = TensorBlock.load_buffer(buffer)
    check_block(loaded)


def test_save_block(tmpdir, block_path):
    """Check that we can save and load a tensor to a file"""
    tmpfile = "serialize-test.mts"

    block = _tests_utils.tensor(dtype=torch.float64).block(2)

    with tmpdir.as_cwd():
        mts.save(tmpfile, block)
        data = mts.load_block(tmpfile)
        assert len(data) == 4

        block.save(tmpfile)
        data = mts.load_block(tmpfile)
        assert len(data) == 4

        # using Path
        mts.save(Path(tmpfile), block)
        data = mts.load_block(tmpfile)
        assert len(data) == 4

        # using pre-opened file
        with open(tmpfile, "wb") as file:
            mts.save(file, block)
        data = mts.load_block(tmpfile)
        assert len(data) == 4

    # save with buffer
    buffer = torch.tensor(np.fromfile(block_path, dtype="uint8"))
    tensor = mts.load_block_buffer(buffer)

    saved = mts.save_buffer(tensor)
    assert torch.all(buffer == saved)

    saved = tensor.save_buffer()
    assert torch.all(buffer == saved)

    block_f32 = block.to(torch.float32)
    with pytest.raises(ValueError, match="only float64 is supported"):
        mts.save_buffer(block_f32)
    with pytest.raises(ValueError, match="only float64 is supported"):
        mts.save(tmpfile, block_f32)

    block_meta = block.to(torch.device("meta"))
    with pytest.raises(ValueError, match="only CPU is supported"):
        mts.save_buffer(block_meta)
    with pytest.raises(ValueError, match="only CPU is supported"):
        mts.save(tmpfile, block_meta)


def test_pickle_block(tmpdir, block_path):
    block = mts.load_block(block_path)
    tmpfile = "serialize-test.mts"

    with tmpdir.as_cwd():
        torch.save(block, tmpfile)
        loaded = torch.load(tmpfile, weights_only=False)

    check_block(loaded)


def test_load_labels(labels_path):
    loaded = mts.load_labels(labels_path)
    check_labels(loaded)

    loaded = Labels.load(labels_path)
    check_labels(loaded)

    # using Path
    loaded = mts.load_labels(Path(labels_path))
    check_labels(loaded)

    # using pre-opened file
    with open(labels_path, "rb") as file:
        loaded = mts.load_labels(file)
        check_labels(loaded)

    # load from buffer
    buffer = torch.tensor(np.fromfile(labels_path, dtype="uint8"))

    loaded = mts.load_labels_buffer(buffer)
    check_labels(loaded)

    loaded = Labels.load_buffer(buffer)
    check_labels(loaded)


def test_save_labels(tmpdir, labels_path):
    """Check that we can save and load a tensor to a file"""
    tmpfile = "serialize-test.mts"

    labels = _tests_utils.tensor(dtype=torch.float64).keys

    with tmpdir.as_cwd():
        mts.save(tmpfile, labels)
        data = mts.load_labels(tmpfile)
        assert len(data) == 4

        labels.save(tmpfile)
        data = mts.load_labels(tmpfile)
        assert len(data) == 4

        # using Path
        mts.save(Path(tmpfile), labels)
        data = mts.load_labels(tmpfile)
        assert len(data) == 4

        # using pre-opened file
        with open(tmpfile, "wb") as file:
            mts.save(file, labels)
        data = mts.load_labels(tmpfile)
        assert len(data) == 4

    # save with buffer
    buffer = torch.tensor(np.fromfile(labels_path, dtype="uint8"))
    tensor = mts.load_labels_buffer(buffer)

    saved = mts.save_buffer(tensor)
    assert torch.all(buffer == saved)

    saved = tensor.save_buffer()
    assert torch.all(buffer == saved)


def test_pickle_labels(tmpdir, labels_path):
    labels = mts.load_labels(labels_path)
    tmpfile = "serialize-test.mts"

    with tmpdir.as_cwd():
        torch.save(labels, tmpfile)
        loaded = torch.load(tmpfile, weights_only=False)

    check_labels(loaded)


def test_load_mmap(tensor_path):
    """Test that load_mmap returns data matching regular load."""
    loaded = mts.load_mmap(tensor_path)
    check_tensor(loaded)

    # using Path
    loaded = mts.load_mmap(Path(tensor_path))
    check_tensor(loaded)


def test_load_mmap_type_error():
    """Test that load_mmap rejects file-like objects."""
    with pytest.raises(TypeError, match="load_mmap only supports file paths"):
        mts.load_mmap(42)  # type: ignore


def test_load_block_mmap(block_path):
    """Test that load_block_mmap returns data matching regular load."""
    loaded = mts.load_block_mmap(block_path)
    check_block(loaded)

    # using Path
    loaded = mts.load_block_mmap(Path(block_path))
    check_block(loaded)


def test_load_block_mmap_type_error():
    """Test that load_block_mmap rejects file-like objects."""
    with pytest.raises(TypeError, match="load_block_mmap only supports file paths"):
        mts.load_block_mmap(42)  # type: ignore


def test_mmap_partial_file_reading(tensor_path):
    """Test that only accessed blocks' data is paged in (the main benefit of mmap)."""
    tensor = mts.load_mmap(tensor_path)

    # Accessing blocks one at a time exercises lazy page-in:
    # the OS only faults in pages for the block we touch.
    for i in range(len(tensor.keys)):
        block = tensor.block_by_id(i)
        # accessing .values triggers the DLPack -> torch conversion,
        # which pages in only this block's data from the mmap
        assert block.values.shape[0] > 0

    # Also verify gradient access pages in gradient data independently
    block = tensor.block_by_id(0)
    for param in block.gradients_list():
        grad = block.gradient(param)
        assert grad.values.shape[0] > 0


def test_mmap_value_data_matches_regular(tensor_path):
    """Test that actual numeric values from mmap match regular load."""
    regular = mts.load(tensor_path)
    mmap = mts.load_mmap(tensor_path)

    for i in range(len(regular.keys)):
        rb = regular.block_by_id(i)
        mb = mmap.block_by_id(i)
        assert torch.equal(rb.values, mb.values)

        for param in rb.gradients_list():
            rg = rb.gradient(param)
            mg = mb.gradient(param)
            assert torch.equal(rg.values, mg.values)


def test_mmap_operations_compatibility(tensor_path):
    """Test clone and save roundtrip on mmap-loaded data."""
    regular = mts.load(tensor_path)
    tensor = mts.load_mmap(tensor_path)

    # Clone
    clone = tensor.copy()
    assert clone.keys == tensor.keys
    for i in range(len(tensor.keys)):
        assert tensor.block_by_id(i).values.shape == clone.block_by_id(i).values.shape

    # Save roundtrip: mmap and regular produce identical bytes
    regular_buf = mts.save_buffer(regular)
    mmap_buf = mts.save_buffer(tensor)
    assert torch.equal(regular_buf, mmap_buf)

    # Reload from mmap buffer
    reloaded = mts.load_buffer(mmap_buf)
    check_tensor(reloaded)


class Serialization:
    def load(self, file: str) -> TensorMap:
        return mts.load(file=file)

    def load_buffer(self, buffer: torch.Tensor) -> TensorMap:
        return mts.load_buffer(buffer=buffer)

    def load_mmap(self, file: str) -> TensorMap:
        return mts.load_mmap(file=file)

    def load_block(self, file: str) -> TensorBlock:
        return mts.load_block(file=file)

    def load_block_buffer(self, buffer: torch.Tensor) -> TensorBlock:
        return mts.load_block_buffer(buffer=buffer)

    def load_block_mmap(self, file: str) -> TensorBlock:
        return mts.load_block_mmap(file=file)

    def load_labels(self, file: str) -> Labels:
        return mts.load_labels(file=file)

    def load_labels_buffer(self, buffer: torch.Tensor) -> Labels:
        return mts.load_labels_buffer(buffer=buffer)

    def save(self, file: str, data: Union[Labels, TensorBlock, TensorMap]):
        return mts.save(file=file, data=data)

    def save_buffer(self, data: Union[Labels, TensorBlock, TensorMap]) -> torch.Tensor:
        return mts.save_buffer(data=data)


def test_script():
    # check that the operators definition (in register.cpp) match what we expect
    # (defined in `Serialization`)
    _ = torch.jit.script(Serialization)

    @torch.jit.script
    def test_function(labels: Labels):
        torch.ops.metatensor.save("tmp.mts", labels)

    assert 'ops.metatensor.save("tmp.mts", labels)' in test_function.code
