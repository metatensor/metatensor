import io

import torch
from packaging import version

import metatensor.torch

from .data import load_data


def check_operation(detach):
    # this only runs basic checks functionality checks, and that the code produces
    # output with the right type
    tensor = load_data("qm7-power-spectrum.npz")
    tensor = metatensor.torch.requires_grad(tensor)

    assert isinstance(tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor._type().name() == "TensorMap"

    for block in tensor:
        assert block.values.requires_grad

    tensor = detach(tensor)

    assert isinstance(tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor._type().name() == "TensorMap"

    for block in tensor:
        assert not block.values.requires_grad


def test_operation_as_python():
    check_operation(metatensor.torch.detach)


def test_operation_as_torch_script():
    scripted = torch.jit.script(metatensor.torch.detach)

    check_operation(scripted)


def test_save_load():
    scripted = torch.jit.script(metatensor.torch.detach)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
