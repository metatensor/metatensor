import torch
from packaging import version

import metatensor.torch

from .data import load_data


def check_operation(join):
    tensor = load_data("qm7-power-spectrum.npz")
    joined_tensor = join([tensor, tensor], axis="properties")

    assert isinstance(joined_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert joined_tensor._type().name() == "TensorMap"

    # test keys
    assert joined_tensor.keys == tensor.keys

    # test property names
    names = tensor.block(0).properties.names
    assert joined_tensor.block(0).properties.names == ["tensor"] + names

    # test samples
    assert joined_tensor.block(0).samples == tensor.block(0).samples

    # test if gradients exist
    assert sorted(joined_tensor[0].gradients_list()) == sorted(
        tensor[0].gradients_list()
    )


def test_operations_as_python():
    check_operation(metatensor.torch.join)


def test_operations_as_torch_script():
    scripted = torch.jit.script(metatensor.torch.join)
    check_operation(scripted)
