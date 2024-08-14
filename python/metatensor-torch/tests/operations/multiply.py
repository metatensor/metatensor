import io
import os

import torch

import metatensor.torch


def test_multiply():
    tensor = metatensor.torch.load(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "metatensor-operations",
            "tests",
            "data",
            "qm7-power-spectrum.npz",
        )
    )
    product_tensor = metatensor.torch.multiply(tensor, tensor)
    assert metatensor.torch.equal_metadata(product_tensor, tensor)
    assert torch.allclose(product_tensor.block(0).values, tensor.block(0).values ** 2)


def test_save():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.multiply, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
