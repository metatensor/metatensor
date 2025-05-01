import io
import os

import pytest
import torch

import metatensor.torch as mts


def test_subtract():
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
    difference_tensor = mts.subtract(tensor, tensor)
    assert mts.equal_metadata(difference_tensor, tensor)
    assert torch.allclose(
        difference_tensor.block(0).values,
        torch.zeros_like(difference_tensor.block(0).values),
    )


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.subtract, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
