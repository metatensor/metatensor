import io
import os

import torch

import metatensor.torch


def test_add():
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
    sum_tensor = metatensor.torch.add(tensor, tensor)
    assert metatensor.torch.equal_metadata(sum_tensor, tensor)
    assert metatensor.torch.allclose(sum_tensor, metatensor.torch.multiply(tensor, 2))


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.add, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
