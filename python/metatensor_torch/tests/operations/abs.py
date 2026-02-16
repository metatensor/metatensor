import io
import os

import pytest
import torch

import metatensor.torch as mts


def test_abs():
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
    abs_tensor = mts.abs(tensor)

    # check output type
    assert isinstance(abs_tensor, torch.ScriptObject)
    assert abs_tensor._type().name() == "TensorMap"

    # check metadata
    assert mts.equal_metadata(abs_tensor, tensor)

    # check values
    for block in abs_tensor.blocks():
        assert torch.all(block.values >= 0.0)


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.sum_over_samples, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
