import io
import os

import pytest
import torch

import metatensor.torch as mts


def test_allclose():
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

    assert mts.allclose(tensor, tensor)

    mts.allclose_raise(tensor, tensor)

    assert mts.allclose_block(tensor.block(0), tensor.block(0))

    mts.allclose_block_raise(tensor.block(0), tensor.block(0))


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.allclose, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(mts.allclose_raise, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(mts.allclose_block, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(mts.allclose_block_raise, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
