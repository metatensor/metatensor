import torch
from packaging import version

import equistore.torch

from .data import load_data


def check_operation(allclose):
    # this only runs basic checks functionality checks, and that the code produces
    # output with the right type
    tensor1 = load_data("qm7-power-spectrum.npz")
    tensor2 = load_data("qm7-power-spectrum.npz")

    assert isinstance(tensor1, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor1._type().name() == "TensorMap"

    assert allclose(tensor1, tensor2)


def test_operation_as_python():
    check_operation(equistore.torch.allclose)


def test_operation_as_torch_script():
    scripted = torch.jit.script(equistore.torch.allclose)

    check_operation(scripted)
