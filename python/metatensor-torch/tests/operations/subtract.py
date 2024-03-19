import io

import torch

import metatensor.torch

from ._data import load_data


def test_subtract():
    tensor = load_data("qm7-power-spectrum.npz")
    difference_tensor = metatensor.torch.subtract(tensor, tensor)
    assert metatensor.torch.equal_metadata(difference_tensor, tensor)
    assert torch.allclose(
        difference_tensor.block(0).values,
        torch.zeros_like(difference_tensor.block(0).values),
    )


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.subtract, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
