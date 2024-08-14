import io
import os

import torch

import metatensor.torch


def test_allclose():
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

    assert metatensor.torch.allclose(tensor, tensor)

    metatensor.torch.allclose_raise(tensor, tensor)

    assert metatensor.torch.allclose_block(tensor.block(0), tensor.block(0))

    metatensor.torch.allclose_block_raise(tensor.block(0), tensor.block(0))


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.allclose, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.allclose_raise, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.allclose_block, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.allclose_block_raise, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
