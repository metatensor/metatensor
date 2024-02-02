import io

import torch
from packaging import version

import metatensor.torch

from .data import load_data


def check_operation(zeros_like):
    tensor = load_data("qm7-power-spectrum.npz")
    zero_tensor = zeros_like(tensor)

    # right output type
    assert isinstance(zero_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert zero_tensor._type().name() == "TensorMap"

    # right metadata
    assert metatensor.torch.equal_metadata(zero_tensor, tensor)

    # right values
    for block in zero_tensor.blocks():
        assert torch.all(block.values == 0)


def test_operation_as_python():
    check_operation(metatensor.torch.zeros_like)


def test_operation_as_torch_script():
    scripted = torch.jit.script(metatensor.torch.zeros_like)
    check_operation(scripted)


def test_save_load():
    scripted = torch.jit.script(metatensor.torch.zeros_like)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
