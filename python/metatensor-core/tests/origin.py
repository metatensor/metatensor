import numpy as np
import pytest
import torch

from metatensor import Labels, MetatensorError, TensorBlock, TensorMap


def test_different_origins():
    """
    Test that an error is thrown when attempting to initialize a TensorMap with
    TensorBlocks with different origins
    """

    keys = Labels.range("dummy", 2)

    block_numpy = TensorBlock(
        values=np.array([[0.0]]),
        samples=Labels.single(),
        components=[],
        properties=Labels.single(),
    )

    block_torch = TensorBlock(
        values=torch.tensor([[0.0]]),
        samples=Labels.single(),
        components=[],
        properties=Labels.single(),
    )

    with pytest.raises(MetatensorError, match="different origins"):
        TensorMap(keys=keys, blocks=[block_numpy, block_torch])
