import io

import torch

import metatensor.torch

from ._data import load_data


def get_tensor_map():
    # Since there will be different samples in blocks with different species_center,
    # and this would make all the operations fail due to different samples in different
    # blocks, we build a TensorMap with only two blocks which have the same samples.
    tensor = load_data("qm7-power-spectrum.npz")
    keys = metatensor.torch.Labels.range("block", 2)
    blocks = [
        tensor.block(
            {"species_center": 6, "species_neighbor_1": 1, "species_neighbor_2": 1}
        ),
        tensor.block(
            {"species_center": 6, "species_neighbor_1": 6, "species_neighbor_2": 6}
        ),
    ]
    return metatensor.torch.TensorMap(keys, blocks)


def test_append():
    tensor = get_tensor_map()

    assert tensor.sample_names == ["structure", "center"]
    new_tensor = metatensor.torch.append_dimension(
        tensor, "samples", "center_2", tensor.block(0).samples.column("center")
    )

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.sample_names == ["structure", "center", "center_2"]


def test_insert():
    tensor = get_tensor_map()

    assert tensor.sample_names == ["structure", "center"]
    new_tensor = metatensor.torch.insert_dimension(
        tensor, "samples", 1, "center_2", tensor.block(0).samples.column("center")
    )

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.sample_names == ["structure", "center_2", "center"]


def test_permute():
    tensor = get_tensor_map()

    assert tensor.sample_names == ["structure", "center"]
    new_tensor = metatensor.torch.permute_dimensions(tensor, "samples", [1, 0])

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.sample_names == ["center", "structure"]


def test_remove():
    tensor = get_tensor_map()

    assert tensor.sample_names == ["structure", "center"]
    new_tensor = metatensor.torch.append_dimension(
        tensor, "samples", "center_2", tensor.block(0).samples.column("center")
    )
    new_tensor = metatensor.torch.remove_dimension(new_tensor, "samples", "center_2")

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.sample_names == ["structure", "center"]


def test_rename():
    tensor = get_tensor_map()

    assert tensor.sample_names == ["structure", "center"]
    new_tensor = metatensor.torch.rename_dimension(
        tensor, "samples", "center", "center_2"
    )

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.sample_names == ["structure", "center_2"]


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.append_dimension, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.insert_dimension, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.permute_dimensions, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.remove_dimension, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.rename_dimension, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
