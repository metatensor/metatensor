import io
import os

import pytest
import torch

import metatensor.torch as mts
from metatensor.torch import Labels


def test_drop_blocks():
    # this only runs basic checks functionality checks, and that the code produces
    # output with the right type
    tensor = mts.load(
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

    keys_to_drop = Labels(names=["center_type"], values=torch.tensor([[1], [8]]))

    tensor = mts.drop_blocks(tensor, keys_to_drop)

    # check type
    assert isinstance(tensor, torch.ScriptObject)
    assert tensor._type().name() == "TensorMap"

    # check remaining block
    expected_keys = Labels(names=["center_type"], values=torch.tensor([[6]]))
    assert tensor.keys == expected_keys


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.drop_blocks, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
