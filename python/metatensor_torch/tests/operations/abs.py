import io
import os

import torch
from packaging import version

import metatensor.torch


def test_abs():
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
    abs_tensor = metatensor.torch.abs(tensor)

    # check output type
    assert isinstance(abs_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert abs_tensor._type().name() == "TensorMap"

    # check metadata
    assert metatensor.torch.equal_metadata(abs_tensor, tensor)

    # check values
    for block in abs_tensor.blocks():
        assert torch.all(block.values >= 0.0)


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.sum_over_samples, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
