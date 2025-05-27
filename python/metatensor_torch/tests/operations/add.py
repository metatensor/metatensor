import io
import os

import pytest
import torch

import metatensor
import metatensor.torch as mts


@pytest.fixture
def tensor_path():
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor_operations",
        "tests",
        "data",
        "qm7-power-spectrum.mts",
    )


def test_type_error(tensor_path):
    # using operations from metatensor-core with type from metatensor-torch
    tensor = mts.load(tensor_path)
    error = "`A` must be a metatensor TensorMap, not <class 'torch.ScriptObject'>"
    warning = (
        "Trying to use operations from metatensor with objects from metatensor-torch, "
        "you should use the operation from `metatensor.torch` as well, e.g. "
        r"`metatensor.torch.add\(...\)` instead of `metatensor.add\(...\)`"
    )
    with pytest.raises(TypeError, match=error):
        with pytest.warns(UserWarning, match=warning):
            metatensor.add(tensor, tensor)


def test_add(tensor_path):
    tensor = mts.load(tensor_path)
    sum_tensor = mts.add(tensor, tensor)
    assert mts.equal_metadata(sum_tensor, tensor)
    assert mts.allclose(sum_tensor, mts.multiply(tensor, 2))


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.add, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
