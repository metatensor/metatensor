import io

import torch
from packaging import version

import metatensor.torch

from .data import load_data


def check_operation(abs):
    tensor = load_data("qm7-power-spectrum.npz")
    abs_tensor = abs(tensor)

    # check output type
    assert isinstance(abs_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert abs_tensor._type().name() == "TensorMap"

    # check metadata
    assert metatensor.torch.equal_metadata(abs_tensor, tensor)

    # check values
    for block in abs_tensor.blocks():
        assert torch.all(block.values >= 0.0)


def test_operation_as_python():
    check_operation(metatensor.torch.abs)


def test_operation_as_torch_script():
    check_operation(torch.jit.script(metatensor.torch.abs))


def test_save_load():
    scripted = torch.jit.script(metatensor.torch.sum_over_samples)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
