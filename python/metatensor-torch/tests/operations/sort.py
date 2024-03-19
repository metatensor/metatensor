import io

import torch
from packaging import version

import metatensor.torch

from ._data import load_data


def test_sort():
    # Very minimal test, mainly checking that the code runs
    tensor = load_data("qm7-power-spectrum.npz")
    sorted_tensor = metatensor.torch.sort(tensor)

    # right output type
    assert isinstance(sorted_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert sorted_tensor._type().name() == "TensorMap"


def test_save():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.sort, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
