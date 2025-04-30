import io
import os

import pytest
import torch

import metatensor.torch as mts
from metatensor.torch import Labels, TensorMap


def get_tensor_map():
    # Since there will be different samples in blocks with different center_type,
    # and this would make all the operations fail due to different samples in different
    # blocks, we build a TensorMap with only two blocks which have the same samples.
    tensor = mts.load(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "metatensor_operations",
            "tests",
            "data",
            "qm7-power-spectrum.mts",
        )
    )
    keys = Labels.range("block", 2)
    blocks = [
        tensor.block({"center_type": 6, "neighbor_1_type": 1, "neighbor_2_type": 1}),
        tensor.block({"center_type": 6, "neighbor_1_type": 6, "neighbor_2_type": 6}),
    ]
    return TensorMap(keys, blocks)


def test_append():
    tensor = get_tensor_map()

    assert tensor.sample_names == ["system", "atom"]
    new_tensor = mts.append_dimension(
        tensor, "samples", "center_2", tensor.block(0).samples.column("atom")
    )

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.sample_names == ["system", "atom", "center_2"]

    # using an integer value
    new_tensor = mts.append_dimension(tensor, "properties", "new", 4)
    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.property_names == ["l", "n_1", "n_2", "new"]


def test_insert():
    tensor = get_tensor_map()

    assert tensor.sample_names == ["system", "atom"]
    new_tensor = mts.insert_dimension(
        tensor, "samples", 1, "center_2", tensor.block(0).samples.column("atom")
    )

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.sample_names == ["system", "center_2", "atom"]

    # using an integer value
    new_tensor = mts.insert_dimension(tensor, "properties", 0, "new", 3)

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.property_names == ["new", "l", "n_1", "n_2"]


def test_permute():
    tensor = get_tensor_map()

    assert tensor.sample_names == ["system", "atom"]
    new_tensor = mts.permute_dimensions(tensor, "samples", [1, 0])

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.sample_names == ["atom", "system"]


def test_remove():
    tensor = get_tensor_map()

    assert tensor.sample_names == ["system", "atom"]
    new_tensor = mts.append_dimension(
        tensor, "samples", "center_2", tensor.block(0).samples.column("atom")
    )
    new_tensor = mts.remove_dimension(new_tensor, "samples", "center_2")

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.sample_names == ["system", "atom"]


def test_rename():
    tensor = get_tensor_map()

    assert tensor.sample_names == ["system", "atom"]
    new_tensor = mts.rename_dimension(tensor, "samples", "atom", "center_2")

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.sample_names == ["system", "center_2"]


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.append_dimension, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(mts.insert_dimension, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(mts.permute_dimensions, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(mts.remove_dimension, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(mts.rename_dimension, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
