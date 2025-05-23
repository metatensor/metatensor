import io
import os

import pytest
import torch

import metatensor.torch as mts


def test_ones_like():
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
    ones_tensor = mts.ones_like(tensor)

    # right output type
    assert isinstance(ones_tensor, torch.ScriptObject)
    assert ones_tensor._type().name() == "TensorMap"

    # right metadata
    assert mts.equal_metadata(ones_tensor, tensor)

    # right values
    for block in ones_tensor.blocks():
        assert torch.all(block.values == 1)


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.ones_like, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
