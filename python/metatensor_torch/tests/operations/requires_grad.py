import io
import os

import torch
from packaging import version

import metatensor.torch


def test_requires_grad():
    # this only runs basic checks functionality checks, and that the code produces
    # output with the right type
    tensor = metatensor.torch.load(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "metatensor_operations",
            "tests",
            "data",
            "qm7-power-spectrum.npz",
        )
    )

    tensor = metatensor.torch.requires_grad(tensor)

    assert isinstance(tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor._type().name() == "TensorMap"

    for block in tensor:
        assert block.values.requires_grad

    tensor = metatensor.torch.requires_grad(tensor, False)

    assert isinstance(tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor._type().name() == "TensorMap"

    for block in tensor:
        assert not block.values.requires_grad


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.requires_grad, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
