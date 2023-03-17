import os

import numpy as np
import pytest

import equistore


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


class TestRandomLike:
    @pytest.fixture
    def tensor(self):
        tensor = equistore.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        return tensor

    def test_random_uniform_like_nocomponent(self, tensor):
        rand_tensor = equistore.random_uniform_like(tensor)
        assert equistore.equal_metadata(tensor, rand_tensor)

        for _, rand_block in rand_tensor:
            assert np.all(rand_block.values >= 0)
            assert np.all(rand_block.values <= 1)
            for _, rand_gradient in rand_block.gradients():
                assert np.all(rand_gradient.data >= 0)
                assert np.all(rand_gradient.data <= 1)

    def test_random_uniform_like_component(self, tensor):
        rand_tensor = equistore.random_uniform_like(tensor)
        rand_tensor_positions = equistore.random_uniform_like(
            tensor, parameters="positions"
        )

        assert equistore.equal_metadata(tensor, rand_tensor)
        assert equistore.equal_metadata(tensor, rand_tensor_positions)

        for key, rand_block in rand_tensor:
            rand_block_pos = rand_tensor_positions.block(key)
            assert np.all(rand_block.values >= 0)
            assert np.all(rand_block.values <= 1)
            assert np.all(rand_block_pos.values >= 0)
            assert np.all(rand_block_pos.values <= 1)
            for _, rand_gradient in rand_block.gradients():
                assert np.all(rand_gradient.data >= 0)
                assert np.all(rand_gradient.data <= 1)
            for _, rand_gradient_pos in rand_block_pos.gradients():
                assert np.all(rand_gradient_pos.data >= 0)
                assert np.all(rand_gradient_pos.data <= 1)

    def test_random_uniform_like_error(self, tensor):
        msg = (
            "The requested parameter 'err' in random_uniform_like_block "
            + "is not a valid parameterfor the TensorBlock"
        )
        with pytest.raises(TypeError, match=msg):
            tensor = equistore.random_uniform_like(
                tensor, parameters=["positions", "err"]
            )
