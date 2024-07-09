import io

import torch

import metatensor.torch


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.is_contiguous, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.make_contiguous, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
