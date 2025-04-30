import io
import os

import pytest
import torch

import metatensor.torch as mts


def test_zeros_like():
    tensor = mts.load(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "metatensor_operations",
            "tests",
            "data",
            "qm7-power-spectrum.mts",
        )
    )
    zero_tensor = mts.zeros_like(tensor)

    # right output type
    assert isinstance(zero_tensor, torch.ScriptObject)
    assert zero_tensor._type().name() == "TensorMap"

    # right metadata
    assert mts.equal_metadata(zero_tensor, tensor)

    # right values
    for block in zero_tensor.blocks():
        assert torch.all(block.values == 0)


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.zeros_like, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
