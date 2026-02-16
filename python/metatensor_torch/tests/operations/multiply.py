import io
import os

import pytest
import torch

import metatensor.torch as mts


def test_multiply():
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
    product_tensor = mts.multiply(tensor, tensor)
    assert mts.equal_metadata(product_tensor, tensor)
    assert torch.allclose(product_tensor.block(0).values, tensor.block(0).values ** 2)


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.multiply, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
