import io

import torch

import metatensor.torch

from ._data import load_data


def test_add():
    tensor = load_data("qm7-power-spectrum.npz")
    sum_tensor = metatensor.torch.add(tensor, tensor)
    assert metatensor.torch.equal_metadata(sum_tensor, tensor)
    assert metatensor.torch.allclose(sum_tensor, metatensor.torch.multiply(tensor, 2))


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.add, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
