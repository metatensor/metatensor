import io

import torch
from packaging import version

import metatensor.torch
from metatensor.torch import Labels

from .data import load_data


def check_operation(drop_blocks):
    # this only runs basic checks functionality checks, and that the code produces
    # output with the right type
    tensor = load_data("qm7-power-spectrum.npz")
    tensor = tensor.keys_to_properties("species_neighbor_1")
    tensor = tensor.keys_to_properties("species_neighbor_2")

    assert tensor.keys == Labels(
        names=["species_center"], values=torch.tensor([[1], [6], [8]])
    )

    keys_to_drop = Labels(names=["species_center"], values=torch.tensor([[1], [8]]))

    tensor = drop_blocks(tensor, keys_to_drop)

    # check type
    assert isinstance(tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor._type().name() == "TensorMap"

    # check remaining block
    expected_keys = Labels(names=["species_center"], values=torch.tensor([[6]]))
    assert tensor.keys == expected_keys


def test_operation_as_python():
    check_operation(metatensor.torch.drop_blocks)


def test_operation_as_torch_script():
    check_operation(torch.jit.script(metatensor.torch.drop_blocks))


def test_save_load():
    scripted = torch.jit.script(metatensor.torch.drop_blocks)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
