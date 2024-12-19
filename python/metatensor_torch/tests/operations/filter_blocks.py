import io
import os

import torch
from packaging import version

import metatensor.torch
from metatensor.torch import Labels


def test_filter_blocks():
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
            "qm7-power-spectrum.mts",
        )
    )
    tensor = tensor.keys_to_properties("neighbor_1_type")
    tensor = tensor.keys_to_properties("neighbor_2_type")

    assert tensor.keys == Labels(
        names=["center_type"], values=torch.tensor([[1], [6], [8]])
    )

    keys_to_keep = Labels(names=["center_type"], values=torch.tensor([[1], [8]]))

    tensor = metatensor.torch.filter_blocks(tensor, keys_to_keep)

    # check type
    assert isinstance(tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor._type().name() == "TensorMap"

    # check remaining block
    expected_keys = Labels(names=["center_type"], values=torch.tensor([[1], [8]]))
    assert tensor.keys == expected_keys


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.filter_blocks, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
