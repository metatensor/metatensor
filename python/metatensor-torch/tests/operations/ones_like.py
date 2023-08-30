import torch
from packaging import version

import metatensor.torch

from .data import load_data


def check_operation(ones_like):
    tensor = load_data("qm7-power-spectrum.npz")
    ones_tensor = ones_like(tensor)

    # right output type
    assert isinstance(ones_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert ones_tensor._type().name() == "TensorMap"

    # right metadata
    assert metatensor.torch.equal_metadata(ones_tensor, tensor)

    # right values
    for block in ones_tensor.blocks():
        assert torch.all(block.values == 1)


def test_operation_as_python():
    check_operation(metatensor.torch.ones_like)


def test_operation_as_torch_script():
    scripted = torch.jit.script(metatensor.torch.ones_like)
    check_operation(scripted)
