import io
import os

import torch

import metatensor.torch


def test_pow():
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
    power_tensor = metatensor.torch.pow(tensor, 2.0)
    assert metatensor.torch.equal_metadata(power_tensor, tensor)
    assert metatensor.torch.allclose(
        power_tensor, metatensor.torch.multiply(tensor, tensor)
    )


def test_save():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.pow, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
