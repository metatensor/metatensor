import io
import os

import pytest
import torch

import metatensor.torch as mts


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_block_from_array(device):
    values = torch.arange(42, device=device).reshape(2, 3, 7).to(torch.float64)
    block = mts.block_from_array(values)

    # check output type
    assert isinstance(block, torch.ScriptObject)
    assert block._type().name() == "TensorBlock"

    # check values
    if device != "meta":
        assert torch.equal(block.values, values)


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.block_from_array, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
