import os

import pytest

import metatensor as mts


torch = pytest.importorskip("torch")

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_requires_grad():
    tensor = mts.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.mts"))
    tensor = tensor.to(arrays="torch")

    for block in tensor:
        assert not block.values.requires_grad

    tensor_no_grad = tensor
    tensor = mts.requires_grad(tensor_no_grad)

    for block in tensor:
        assert block.values.requires_grad

    # check that the argument was not modified
    for block in tensor_no_grad:
        assert not block.values.requires_grad

    tensor = mts.requires_grad(tensor, requires_grad=False)
    for block in tensor:
        assert not block.values.requires_grad
