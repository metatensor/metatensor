import io

import torch

import metatensor.torch

from ._data import load_data


def test_divide():
    tensor = load_data("qm7-power-spectrum.npz")
    quotient_tensor = metatensor.torch.divide(tensor, tensor)
    assert metatensor.torch.equal_metadata(quotient_tensor, tensor)
    assert torch.allclose(
        torch.nan_to_num(quotient_tensor.block(0).values, 1.0),  # replace nan with 1.0
        torch.ones_like(quotient_tensor.block(0).values),
    )


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.divide, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
