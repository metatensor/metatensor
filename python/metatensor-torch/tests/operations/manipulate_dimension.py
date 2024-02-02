import io

import torch

import metatensor.torch

from .data import load_data


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


def check_append(append):
    tensor = get_tensor_map()

    assert tensor.sample_names == ["structure", "center"]
    new_tensor = append(
        tensor, "samples", "center_2", tensor.block(0).samples.column("center")
    )

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.sample_names == ["structure", "center", "center_2"]


def check_insert(insert):
    tensor = get_tensor_map()

    assert tensor.sample_names == ["structure", "center"]
    new_tensor = insert(
        tensor, "samples", 1, "center_2", tensor.block(0).samples.column("center")
    )

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.sample_names == ["structure", "center_2", "center"]


def check_permute(permute):
    tensor = get_tensor_map()

    assert tensor.sample_names == ["structure", "center"]
    new_tensor = permute(tensor, "samples", [1, 0])

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.sample_names == ["center", "structure"]


def check_remove(remove):
    tensor = get_tensor_map()

    assert tensor.sample_names == ["structure", "center"]
    new_tensor = metatensor.torch.append_dimension(
        tensor, "samples", "center_2", tensor.block(0).samples.column("center")
    )
    new_tensor = remove(new_tensor, "samples", "center_2")

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.sample_names == ["structure", "center"]


def check_rename(rename):
    tensor = get_tensor_map()

    assert tensor.sample_names == ["structure", "center"]
    new_tensor = rename(tensor, "samples", "center", "center_2")

    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor.sample_names == ["structure", "center_2"]


def test_operations_as_python():
    check_append(metatensor.torch.append_dimension)
    check_insert(metatensor.torch.insert_dimension)
    check_permute(metatensor.torch.permute_dimensions)
    check_remove(metatensor.torch.remove_dimension)
    check_rename(metatensor.torch.rename_dimension)


def test_operations_as_torch_script():
    check_append(torch.jit.script(metatensor.torch.append_dimension))
    check_insert(torch.jit.script(metatensor.torch.insert_dimension))
    check_permute(torch.jit.script(metatensor.torch.permute_dimensions))
    check_remove(torch.jit.script(metatensor.torch.remove_dimension))
    check_rename(torch.jit.script(metatensor.torch.rename_dimension))


def test_save_load():
    scripted = torch.jit.script(metatensor.torch.append_dimension)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
    scripted = torch.jit.script(metatensor.torch.insert_dimension)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
    scripted = torch.jit.script(metatensor.torch.permute_dimensions)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
    scripted = torch.jit.script(metatensor.torch.remove_dimension)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
    scripted = torch.jit.script(metatensor.torch.rename_dimension)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
