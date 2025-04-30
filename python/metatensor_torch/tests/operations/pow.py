import io
import os

import pytest
import torch

import metatensor.torch as mts


def test_pow():
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
    power_tensor = mts.pow(tensor, 2.0)
    assert mts.equal_metadata(power_tensor, tensor)
    assert mts.allclose(power_tensor, mts.multiply(tensor, tensor))


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.pow, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
