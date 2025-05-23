import io
import os

import pytest
import torch

import metatensor.torch as mts


def test_empty_like():
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
    empty_tensor = mts.empty_like(tensor)

    # right output type
    assert isinstance(empty_tensor, torch.ScriptObject)
    assert empty_tensor._type().name() == "TensorMap"

    # right metadata
    assert mts.equal_metadata(empty_tensor, tensor)


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.empty_like, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
