import io
import os

import pytest
import torch

import metatensor.torch as mts


def test_dot():
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
    dot_tensor = mts.dot(tensor, mts.remove_gradients(tensor))

    # right output type
    assert isinstance(dot_tensor, torch.ScriptObject)
    assert dot_tensor._type().name() == "TensorMap"

    # right metadata
    for key in tensor.keys:
        assert dot_tensor.block(key).samples == tensor.block(key).samples
        assert dot_tensor.block(key).properties == tensor.block(key).samples

    # right values
    for key in tensor.keys:
        assert torch.allclose(
            dot_tensor.block(key).values,
            tensor.block(key).values @ tensor.block(key).values.T,
        )


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.dot, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
