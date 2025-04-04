import numpy as np
import pytest

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


@pytest.fixture
def tensor():
    # samples are descending, components and properties are ascending
    block_1 = TensorBlock(
        values=np.array(
            [
                [3, 5],
                [1, 2],
            ]
        ),
        samples=Labels(["s"], np.array([[2], [0]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )

    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[8, 3], [9, 4]],
                    [[6, 1], [7, 2]],
                ]
            ),
            samples=Labels(["sample", "g"], np.array([[1, 1], [0, 1]])),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_1.properties,
        ),
    )

    # samples are disordered, components are ascending, properties are descending
    block_2 = TensorBlock(
        values=np.array(
            [
                [3, 4],
                [5, 6],
                [1, 2],
            ]
        ),
        samples=Labels(["s"], np.array([[7], [0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[1], [0]])),
    )
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[15, 14], [11, 10]],
                    [[13, 12], [15, 14]],
                    [[11, 10], [13, 12]],
                ]
            ),
            samples=Labels(
                ["sample", "g"],
                np.array(
                    [
                        [1, 1],
                        [2, 1],
                        [0, 1],
                    ]
                ),
            ),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_2.properties,
        ),
    )
    keys = Labels(names=["key_1", "key_2"], values=np.array([[1, 0], [0, 0]]))
    # block order is descending
    return TensorMap(keys, [block_2, block_1])


@pytest.fixture
def tensor_with_empty_block():
    # samples are descending, components and properties are ascending
    block_1 = TensorBlock(
        values=np.array(
            [
                [3, 5],
                [1, 2],
            ]
        ),
        samples=Labels(["s"], np.array([[2], [0]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )

    block_2 = TensorBlock(
        samples=Labels(["s"], np.array([], dtype=np.int64)),
        components=[],
        properties=Labels.range("p", 1),
        values=np.array([], dtype=np.int64).reshape(-1, 1),
    )

    # samples are disordered, components are ascending, properties are descending
    block_3 = TensorBlock(
        values=np.array(
            [
                [3, 4],
                [5, 6],
                [1, 2],
            ]
        ),
        samples=Labels(["s"], np.array([[7], [0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[1], [0]])),
    )

    keys = Labels(names=["key_1", "key_2"], values=np.array([[1, 1], [0, 1], [1, 0]]))
    # block order is descending
    return TensorMap(keys, [block_1, block_2, block_3])


@pytest.fixture
def tensor_with_empty_block_sorted():
    # samples are descending, components and properties are ascending
    block_1 = TensorBlock(
        values=np.array(
            [
                [1, 2],
                [3, 5],
            ]
        ),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )

    block_2 = TensorBlock(
        samples=Labels(["s"], np.array([], dtype=np.int64)),
        components=[],
        properties=Labels.range("p", 1),
        values=np.array([], dtype=np.int64).reshape(-1, 1),
    )

    # samples are disordered, components are ascending, properties are descending
    block_3 = TensorBlock(
        values=np.array(
            [
                [6, 5],
                [2, 1],
                [4, 3],
            ]
        ),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )

    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 1], [1, 0], [1, 1]]))
    # block order is descending
    return TensorMap(keys, [block_2, block_3, block_1])


def tensor_with_single_empty_block():
    return TensorMap(
        Labels.single(),
        [
            TensorBlock(
                samples=Labels.empty(["_"]),
                components=[],
                properties=Labels.single(),
                values=np.empty((0, 1)),
            )
        ],
    )


@pytest.fixture
def tensor_sorted_ascending():
    """
    This is the `tensor` fixture sorted in ascending order how it should be returned
    when applying metatensor.operations.sort with `descending=False` option.
    """
    block_1 = TensorBlock(
        values=np.array(
            [
                [1, 2],
                [3, 5],
            ]
        ),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )

    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[8, 3], [9, 4]],
                    [[6, 1], [7, 2]],
                ]
            ),
            samples=Labels(
                ["sample", "g"],
                np.array(
                    [
                        [0, 1],
                        [1, 1],
                    ]
                ),
            ),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_1.properties,
        ),
    )
    block_2 = TensorBlock(
        values=np.array(
            [
                [6, 5],
                [2, 1],
                [4, 3],
            ]
        ),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )

    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[14, 15], [10, 11]],
                    [[12, 13], [14, 15]],
                    [[10, 11], [12, 13]],
                ]
            ),
            samples=Labels(
                ["sample", "g"],
                np.array(
                    [
                        [0, 1],
                        [1, 1],
                        [2, 1],
                    ]
                ),
            ),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_2.properties,
        ),
    )

    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
    return TensorMap(keys, [block_1, block_2])


@pytest.fixture
def tensor_sorted_descending():
    """
    This is the `tensor` fixture sorted in descending order how it should be returned
    when applying metatensor.operations.sort with `descending=True` option.
    """
    block_1 = TensorBlock(
        values=np.array(
            [
                [3, 5],
                [1, 2],
            ]
        ),
        samples=Labels(["s"], np.array([[2], [0]])),
        components=[],
        properties=Labels(["p"], np.array([[1], [0]])),
    )

    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[4, 9], [3, 8]],
                    [[2, 7], [1, 6]],
                ]
            ),
            samples=Labels(
                ["sample", "g"],
                np.array(
                    [
                        [1, 1],
                        [0, 1],
                    ]
                ),
            ),
            components=[
                Labels(["c"], np.array([[1], [0]])),
            ],
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=np.array(
            [
                [3, 4],
                [1, 2],
                [5, 6],
            ]
        ),
        samples=Labels(["s"], np.array([[7], [2], [0]])),
        components=[],
        properties=Labels(["p"], np.array([[1], [0]])),
    )
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[11, 10], [15, 14]],
                    [[15, 14], [13, 12]],
                    [[13, 12], [11, 10]],
                ]
            ),
            samples=Labels(
                ["sample", "g"],
                np.array(
                    [
                        [2, 1],
                        [1, 1],
                        [0, 1],
                    ]
                ),
            ),
            components=[
                Labels(["c"], np.array([[1], [0]])),
            ],
            properties=block_2.properties,
        ),
    )
    keys = Labels(
        names=["key_1", "key_2"],
        values=np.array(
            [
                [1, 0],
                [0, 0],
            ]
        ),
    )
    return TensorMap(keys, [block_2, block_1])


def test_sort(tensor, tensor_sorted_ascending):
    metatensor.allclose_block_raise(
        metatensor.sort_block(tensor.block(0)), tensor_sorted_ascending.block(1)
    )
    metatensor.allclose_block_raise(
        metatensor.sort_block(tensor.block(1)), tensor_sorted_ascending.block(0)
    )

    metatensor.allclose_raise(metatensor.sort(tensor), tensor_sorted_ascending)


def test_sort_descending(tensor, tensor_sorted_descending):
    metatensor.allclose_block_raise(
        tensor_sorted_descending.block(0),
        metatensor.sort_block(tensor.block(0), descending=True),
    )
    metatensor.allclose_block_raise(
        tensor_sorted_descending.block(0),
        metatensor.sort_block(tensor.block(0), descending=True),
    )


def test_sort_empty_block(tensor_with_empty_block, tensor_with_empty_block_sorted):
    metatensor.allclose_block_raise(
        metatensor.sort_block(tensor_with_empty_block.block(0)),
        tensor_with_empty_block_sorted.block(2),
    )
    metatensor.allclose_block_raise(
        metatensor.sort_block(tensor_with_empty_block.block(1)),
        tensor_with_empty_block_sorted.block(0),
    )
    metatensor.allclose_block_raise(
        metatensor.sort_block(tensor_with_empty_block.block(2)),
        tensor_with_empty_block_sorted.block(1),
    )
    # Check that this command does not raise an error
    metatensor.sort(tensor_with_single_empty_block())


def test_raise_error(tensor, tensor_sorted_ascending):
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
