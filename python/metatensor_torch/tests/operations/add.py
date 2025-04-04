import io
import os

import pytest
import torch
from packaging import version

import metatensor
import metatensor.torch


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
    tensor = metatensor.torch.load(tensor_path)
    error = "`A` must be a metatensor TensorMap, not <class 'torch.ScriptObject'>"
    warning = (
        "Trying to use operations from metatensor with objects from metatensor-torch, "
        "you should use the operation from `metatensor.torch` as well, e.g. "
        r"`metatensor.torch.add\(...\)` instead of `metatensor.add\(...\)`"
    )
    with pytest.raises(TypeError, match=error):
        if version.parse(torch.__version__) >= version.parse("2.1"):
            with pytest.warns(UserWarning, match=warning):
                metatensor.add(tensor, tensor)
        else:
            # no warning before torch 2.1
            metatensor.add(tensor, tensor)


def test_add(tensor_path):
    tensor = metatensor.torch.load(tensor_path)
    sum_tensor = metatensor.torch.add(tensor, tensor)
    assert metatensor.torch.equal_metadata(sum_tensor, tensor)
    assert metatensor.torch.allclose(sum_tensor, metatensor.torch.multiply(tensor, 2))


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.add, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
