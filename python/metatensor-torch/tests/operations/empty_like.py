import io

import torch
from packaging import version

import metatensor.torch

from ._data import load_data


def test_empty_like():
    tensor = load_data("qm7-power-spectrum.npz")
    empty_tensor = metatensor.torch.empty_like(tensor)

    # right output type
    assert isinstance(empty_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert empty_tensor._type().name() == "TensorMap"

    # right metadata
    assert metatensor.torch.equal_metadata(empty_tensor, tensor)


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.empty_like, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
