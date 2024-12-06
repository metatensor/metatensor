import io

import pytest
import torch
from packaging import version

import metatensor.torch


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_block_from_array(device):
    values = torch.arange(42, device=device).reshape(2, 3, 7).to(torch.float64)
    block = metatensor.torch.block_from_array(values)

    # check output type
    assert isinstance(block, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert block._type().name() == "TensorBlock"

    # check values
    if device != "meta":
        assert torch.equal(block.values, values)


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.block_from_array, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
