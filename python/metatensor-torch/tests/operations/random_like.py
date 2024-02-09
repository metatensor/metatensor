import io

import torch
from packaging import version

import metatensor.torch

from ._data import load_data


def test_random_uniform_like():
    tensor = load_data("qm7-power-spectrum.npz")
    random_tensor = metatensor.torch.random_uniform_like(tensor)

    # right output type
    assert isinstance(random_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert random_tensor._type().name() == "TensorMap"

    # right metadata
    assert metatensor.torch.equal_metadata(random_tensor, tensor)


def test_save():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.random_uniform_like, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
