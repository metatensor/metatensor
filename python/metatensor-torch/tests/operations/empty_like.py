import torch
from packaging import version

import metatensor.torch

from .data import load_data


def check_operation(empty_like):
    tensor = load_data("qm7-power-spectrum.npz")
    empty_tensor = empty_like(tensor)

    # right output type
    assert isinstance(empty_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert empty_tensor._type().name() == "TensorMap"

    # right metadata
    assert metatensor.torch.equal_metadata(empty_tensor, tensor)


def test_operation_as_python():
    check_operation(metatensor.torch.empty_like)


def test_operation_as_torch_script():
    scripted = torch.jit.script(metatensor.torch.empty_like)
    check_operation(scripted)
