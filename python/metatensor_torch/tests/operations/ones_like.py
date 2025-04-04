import io
import os

import torch
from packaging import version

import metatensor.torch


def test_ones_like():
    tensor = metatensor.torch.load(
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
