import io

import torch
from packaging import version

import metatensor.torch

from ._data import load_data


def test_ones_like():
    tensor = load_data("qm7-power-spectrum.npz")
    ones_tensor = metatensor.torch.ones_like(tensor)

    # right output type
    assert isinstance(ones_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert ones_tensor._type().name() == "TensorMap"

    # right metadata
    assert metatensor.torch.equal_metadata(ones_tensor, tensor)

    # right values
    for block in ones_tensor.blocks():
        assert torch.all(block.values == 1)


def test_save():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.ones_like, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
