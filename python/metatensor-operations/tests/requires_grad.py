import os

import pytest

import metatensor


torch = pytest.importorskip("torch")

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_requires_grad():
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        # the npz is using DEFLATE compression, metatensor only supports STORED
        use_numpy=True,
    )
    tensor = metatensor.to(tensor, backend="torch")

    for block in tensor:
        assert not block.values.requires_grad

    tensor_no_grad = tensor
    tensor = metatensor.requires_grad(tensor_no_grad)

    for block in tensor:
        assert block.values.requires_grad

    # check that the argument was not modified
    for block in tensor_no_grad:
        assert not block.values.requires_grad

    tensor = metatensor.requires_grad(tensor, requires_grad=False)
    for block in tensor:
        assert not block.values.requires_grad
