import os

import numpy as np
import pytest

import equistore


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_random_uniform_like():
    tensor = equistore.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        # the npz is using DEFLATE compression, equistore only supports STORED
        use_numpy=True,
    )
    random_tensor = equistore.random_uniform_like(tensor)
    random_tensor_positions = equistore.random_uniform_like(
        tensor, gradients="positions"
    )

    assert equistore.equal_metadata(random_tensor, tensor)

    tensor_no_cell = equistore.remove_gradients(tensor, "cell")
    assert equistore.equal_metadata(random_tensor_positions, tensor_no_cell)

    # check the values
    for random_block in random_tensor:
        assert np.all(random_block.values >= 0)
        assert np.all(random_block.values < 1)

        for _, random_gradient in random_block.gradients():
            assert np.all(random_gradient.values >= 0)
            assert np.all(random_gradient.values < 1)


def test_random_uniform_like_error():
    tensor = equistore.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        # the npz is using DEFLATE compression, equistore only supports STORED
        use_numpy=True,
    )

    message = (
        "requested gradient 'err' in random_uniform_like is not defined in this tensor"
    )
    with pytest.raises(ValueError, match=message):
        tensor = equistore.random_uniform_like(tensor, gradients=["positions", "err"])


# TODO: add tests with torch & torch scripting/tracing
