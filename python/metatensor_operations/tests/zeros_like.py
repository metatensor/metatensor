import os

import numpy as np
import pytest

import metatensor as mts


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_zeros_like():
    tensor = mts.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.mts"))
    zeros_tensor = mts.zeros_like(tensor)
    zeros_tensor_positions = mts.zeros_like(tensor, gradients="positions")

    assert mts.equal_metadata(zeros_tensor, tensor)

    tensor_no_strain = mts.remove_gradients(tensor, "strain")
    assert mts.equal_metadata(zeros_tensor_positions, tensor_no_strain)

    # check the values
    for key, block in tensor.items():
        zeros_block = zeros_tensor[key]

        assert np.all(zeros_block.values == np.zeros_like(block.values))

        for parameter, gradient in block.gradients():
            zeros_gradient = zeros_block.gradient(parameter)
            assert np.all(zeros_gradient.values == np.zeros_like(gradient.values))


def test_zeros_like_error():
    tensor = mts.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.mts"))

    message = "requested gradient 'err' in 'zeros_like' is not defined in this tensor"
    with pytest.raises(ValueError, match=message):
        tensor = mts.zeros_like(tensor, gradients=["positions", "err"])
