import io
import os

import pytest
import torch

import metatensor.torch


def test_random_uniform_like():
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
    random_tensor = metatensor.torch.random_uniform_like(tensor)

    # right output type
    assert isinstance(random_tensor, torch.ScriptObject)
    assert random_tensor._type().name() == "TensorMap"

    # right metadata
    assert metatensor.torch.equal_metadata(random_tensor, tensor)


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.random_uniform_like, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
