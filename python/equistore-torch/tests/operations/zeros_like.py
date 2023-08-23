import torch
from packaging import version

import equistore.torch

from .data import load_data


def check_operation(zeros_like):
    tensor = load_data("qm7-power-spectrum.npz")
    zero_tensor = zeros_like(tensor)

    # right output type
    assert isinstance(zero_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert zero_tensor._type().name() == "TensorMap"

    # right metadata
    assert equistore.torch.equal_metadata(zero_tensor, tensor)

    # right values
    for block in zero_tensor.blocks():
        assert torch.all(block.values == 0)


def test_operation_as_python():
    check_operation(equistore.torch.zeros_like)


def test_operation_as_torch_script():
    scripted = torch.jit.script(equistore.torch.zeros_like)
    check_operation(scripted)
