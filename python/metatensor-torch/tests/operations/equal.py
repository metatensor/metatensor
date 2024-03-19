import io

import torch

import metatensor.torch

from ._data import load_data


def test_equal():
    tensor = load_data("qm7-power-spectrum.npz")
    assert metatensor.torch.equal(tensor, tensor)

    metatensor.torch.equal_raise(tensor, tensor)

    assert metatensor.torch.equal_block(tensor.block(0), tensor.block(0))

    metatensor.torch.equal_block_raise(tensor.block(0), tensor.block(0))


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.equal, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.equal_raise, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.equal_block, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.equal_block_raise, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
