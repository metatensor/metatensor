import io
import os

import pytest
import torch

import metatensor.torch as mts


def test_join():
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

    joined_tensor = mts.join(
        [tensor, tensor], axis="properties", add_dimension="tensor"
    )

    assert isinstance(joined_tensor, torch.ScriptObject)
    assert joined_tensor._type().name() == "TensorMap"

    # test keys
    assert joined_tensor.keys == tensor.keys

    # test property names
    names = tensor.block(0).properties.names
    assert joined_tensor.block(0).properties.names == names + ["tensor"]

    # test samples
    assert joined_tensor.block(0).samples == tensor.block(0).samples

    # test if gradients exist
    assert sorted(joined_tensor[0].gradients_list()) == sorted(
        tensor[0].gradients_list()
    )


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.join, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
