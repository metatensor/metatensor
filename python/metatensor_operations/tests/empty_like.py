import os

import pytest

import metatensor as mts


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_empty_like():
    tensor = mts.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.mts"))
    empty_tensor = mts.empty_like(tensor)
    empty_tensor_positions = mts.empty_like(tensor, gradients="positions")

    assert mts.equal_metadata(empty_tensor, tensor)

    tensor_no_strain = mts.remove_gradients(tensor, "strain")
    assert mts.equal_metadata(empty_tensor_positions, tensor_no_strain)


def test_empty_like_error():
    tensor = mts.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.mts"))

    message = "requested gradient 'err' in 'empty_like' is not defined in this tensor"
    with pytest.raises(ValueError, match=message):
        tensor = mts.empty_like(tensor, gradients=["positions", "err"])
