import torch
from packaging import version

import equistore.torch

from .data import load_data


def check_operation(remove_gradients):
    # this only runs basic checks functionality checks, and that the code produces
    # output with the right type
    tensor = load_data("qm7-power-spectrum.npz")

    assert isinstance(tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor._type().name() == "TensorMap"

    assert set(tensor.block(0).gradients_list()) == set(["cell", "positions"])

    tensor = remove_gradients(tensor, ["positions"])

    assert isinstance(tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor._type().name() == "TensorMap"

    assert tensor.block(0).gradients_list() == ["cell"]


def test_operation_as_python():
    check_operation(equistore.torch.remove_gradients)


def test_operation_as_torch_script():
    scripted = torch.jit.script(equistore.torch.remove_gradients)

    check_operation(scripted)
