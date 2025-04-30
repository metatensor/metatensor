import io
import os

import pytest
import torch

import metatensor.torch as mts


def test_divide():
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
    quotient_tensor = mts.divide(tensor, tensor)
    assert mts.equal_metadata(quotient_tensor, tensor)
    assert torch.allclose(
        torch.nan_to_num(quotient_tensor.block(0).values, 1.0),  # replace nan with 1.0
        torch.ones_like(quotient_tensor.block(0).values),
    )


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.divide, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
