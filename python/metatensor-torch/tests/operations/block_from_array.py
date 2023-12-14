import pytest
import torch
from packaging import version

import metatensor.torch


def check_operation(block_from_array, device):
    values = torch.arange(42, device=device).reshape(2, 3, 7).to(torch.float64)
    block = block_from_array(values)

    # check output type
    assert isinstance(block, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert block._type().name() == "TensorBlock"

    # check values
    if device != "meta":
        assert torch.equal(block.values, values)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_operation_as_python(device):
    check_operation(metatensor.torch.block_from_array, device)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_operation_as_torch_script(device):
    check_operation(torch.jit.script(metatensor.torch.block_from_array), device)
