import io
import os

import torch
from packaging import version

import metatensor.torch


def test_join():
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
    joined_tensor = metatensor.torch.join([tensor, tensor], axis="properties")

    assert isinstance(joined_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert joined_tensor._type().name() == "TensorMap"

    # test keys
    assert joined_tensor.keys == tensor.keys

    # test property names
    names = tensor.block(0).properties.names
    assert joined_tensor.block(0).properties.names == ["tensor"] + names

    # test samples
    assert joined_tensor.block(0).samples == tensor.block(0).samples

    # test if gradients exist
    assert sorted(joined_tensor[0].gradients_list()) == sorted(
        tensor[0].gradients_list()
    )


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.join, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
