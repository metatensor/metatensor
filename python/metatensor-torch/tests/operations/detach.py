import torch
from packaging import version

import metatensor.torch

from .data import load_data


def check_operation(detach):
    # this only runs basic checks functionality checks, and that the code produces
    # output with the right type
    tensor = load_data("qm7-power-spectrum.npz")
    for block in tensor:
        block.values.requires_grad_(True)

    assert isinstance(tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor._type().name() == "TensorMap"

    for block in tensor:
        assert block.values.requires_grad

    assert set(tensor.block(0).gradients_list()) == set(["cell", "positions"])

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
