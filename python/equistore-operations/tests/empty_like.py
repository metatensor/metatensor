import os

import pytest

import equistore


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_empty_like():
    tensor = equistore.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        # the npz is using DEFLATE compression, equistore only supports STORED
        use_numpy=True,
    )
    empty_tensor = equistore.empty_like(tensor)
    empty_tensor_positions = equistore.empty_like(tensor, gradients="positions")

    assert equistore.equal_metadata(empty_tensor, tensor)

    tensor_no_cell = equistore.remove_gradients(tensor, "cell")
    assert equistore.equal_metadata(empty_tensor_positions, tensor_no_cell)


def test_empty_like_error():
    tensor = equistore.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        # the npz is using DEFLATE compression, equistore only supports STORED
        use_numpy=True,
    )

    message = "requested gradient 'err' in empty_like is not defined in this tensor"
    with pytest.raises(ValueError, match=message):
        tensor = equistore.empty_like(tensor, gradients=["positions", "err"])


# TODO: add tests with torch & torch scripting/tracing
