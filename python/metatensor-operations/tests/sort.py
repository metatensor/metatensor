import numpy as np
import pytest

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


@pytest.fixture
def tensor():
    # swapped samples from correct order in block_1
    block_1 = TensorBlock(
        values=np.array([[3, 5], [1, 2]]),
        samples=Labels(["s"], np.array([[2], [0]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )

    # fmt: off
    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array([[[8, 3],
                              [9, 4]],
                             [[6, 1],
                              [7, 2]]]),
            samples=Labels(["sample", "g"], np.array([[1, 1], [0, 1]])),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_1.properties,
        ),
    )
    # fmt: on

    # swapped components from correct order in block_1
    block_2 = TensorBlock(
        values=np.array([[2, 1], [4, 3], [6, 5]]),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels(["p"], np.array([[1], [0]])),
    )
    # fmt: off
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [[[11, 10],
                  [13, 12]],
                 [[15, 14],
                  [11, 10]],
                 [[13, 12],
                  [15, 14]]]
            ),
            samples=Labels(
                ["sample", "g"],
                np.array([[0, 1], [1, 1], [2, 1]]),
            ),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_2.properties,
        ),
    )
    # fmt: on
    keys = Labels(names=["key_1", "key_2"], values=np.array([[1, 0], [0, 0]]))
    return TensorMap(keys, [block_2, block_1])


@pytest.fixture
def tensor_sorted():
    block_1 = TensorBlock(
        values=np.array([[1, 2], [3, 5]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )

    # fmt: off
    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array([[[6, 1],
                              [7, 2]],
                             [[8, 3],
                              [9, 4]]]),
            samples=Labels(["sample", "g"], np.array([[0, 1], [1, 1]])),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_1.properties,
        ),
    )
    # fmt: on
    block_2 = TensorBlock(
        values=np.array([[1, 2], [3, 4], [5, 6]]),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )

    # fmt: off
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array([[[10, 11],
                              [12, 13]],
                             [[14, 15],
                              [10, 11]],
                             [[12, 13],
                              [14, 15]]]),
            samples=Labels(
                ["sample", "g"],
                np.array([[0, 1], [1, 1], [2, 1]]),
            ),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_2.properties,
        ),
    )
    # fmt: on

    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
    return TensorMap(keys, [block_1, block_2])


def test_sort(tensor, tensor_sorted):
    metatensor.allclose_block_raise(
        metatensor.sort_block(tensor.block(0)), tensor_sorted.block(1)
    )
    metatensor.allclose_block_raise(
        metatensor.sort_block(tensor.block(1)), tensor_sorted.block(0)
    )

    metatensor.allclose_raise(metatensor.sort(tensor), tensor_sorted)


def test_sort_descending(tensor, tensor_sorted):
    metatensor.allclose_block_raise(
        tensor.block(0),
        metatensor.sort_block(
            tensor_sorted.block(1), axes="properties", descending=True
        ),
    )
    metatensor.allclose_block_raise(
        tensor.block(1),
        metatensor.sort_block(tensor_sorted.block(0), axes="samples", descending=True),
    )


def test_raise_error(tensor, tensor_sorted):
    error_message = (
        "axes` must be one of 'samples', 'components' or 'properties', not 'error'"
    )
    with pytest.raises(ValueError, match=error_message):
        metatensor.operations.sort(tensor, axes="error")

    error_message = (
        "`axes` must be one of 'samples', 'components' or 'properties', not '5'"
    )
    with pytest.raises(ValueError, match=error_message):
        metatensor.operations.sort(tensor, axes=[5])
