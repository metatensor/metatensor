import io
import os

import pytest
import torch

import metatensor.torch as mts


def test_remove_gradients():
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

    assert isinstance(tensor, torch.ScriptObject)
    assert tensor._type().name() == "TensorMap"

    assert set(tensor.block(0).gradients_list()) == set(["strain", "positions"])

    tensor = mts.remove_gradients(tensor, ["positions"])

    assert isinstance(tensor, torch.ScriptObject)
    assert tensor._type().name() == "TensorMap"

    assert tensor.block(0).gradients_list() == ["strain"]


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.remove_gradients, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
