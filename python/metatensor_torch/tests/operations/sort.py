import io
import os

import pytest
import torch
from packaging import version

import metatensor.torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def test_sort():
    # Very minimal test, mainly checking that the code runs
    tensor = metatensor.torch.load(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "metatensor_operations",
            "tests",
            "data",
            "qm7-power-spectrum.mts",
        )
    )
    sorted_tensor = metatensor.torch.sort(tensor)

    # right output type
    assert isinstance(sorted_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert sorted_tensor._type().name() == "TensorMap"


def test_save():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.sort, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)


@pytest.fixture
def tensor():
    # samples are descending, components and properties are ascending
    block_1 = TensorBlock(
        values=torch.tensor([[3, 5], [1, 2]]),
        samples=Labels(["s"], torch.tensor([[2], [0]])),
        components=[],
        properties=Labels(["p"], torch.tensor([[0], [1]])),
    )

    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.tensor([[[8, 3], [9, 4]], [[6, 1], [7, 2]]]),
            samples=Labels(["sample", "g"], torch.tensor([[1, 1], [0, 1]])),
            components=[
                Labels(["c"], torch.tensor([[0], [1]])),
            ],
            properties=block_1.properties,
        ),
    )

    # samples are disordered, components are ascending, properties are descending
    block_2 = TensorBlock(
        values=torch.tensor([[3, 4], [5, 6], [1, 2]]),
        samples=Labels(["s"], torch.tensor([[7], [0], [2]])),
        components=[],
        properties=Labels(["p"], torch.tensor([[1], [0]])),
    )
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.tensor(
                [[[15, 14], [11, 10]], [[13, 12], [15, 14]], [[11, 10], [13, 12]]]
            ),
            samples=Labels(
                ["sample", "g"],
                torch.tensor([[1, 1], [2, 1], [0, 1]]),
            ),
            components=[
                Labels(["c"], torch.tensor([[0], [1]])),
            ],
            properties=block_2.properties,
        ),
    )
    keys = Labels(names=["key_1", "key_2"], values=torch.tensor([[1, 0], [0, 0]]))
    # block order is descending
    return TensorMap(keys, [block_2, block_1])


@pytest.fixture
def tensor_sorted_ascending():
    """
    This is the `tensor` fixture sorted in ascending order how it should be returned
    when applying metatensor.operations.sort with `descending=False` option.
    """
    block_1 = TensorBlock(
        values=torch.tensor([[1, 2], [3, 5]]),
        samples=Labels(["s"], torch.tensor([[0], [2]])),
        components=[],
        properties=Labels(["p"], torch.tensor([[0], [1]])),
    )

    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.tensor([[[8, 3], [9, 4]], [[6, 1], [7, 2]]]),
            samples=Labels(
                ["sample", "g"],
                torch.tensor([[0, 1], [1, 1]]),
            ),
            components=[
                Labels(["c"], torch.tensor([[0], [1]])),
            ],
            properties=block_1.properties,
        ),
    )
    block_2 = TensorBlock(
        values=torch.tensor([[6, 5], [2, 1], [4, 3]]),
        samples=Labels(["s"], torch.tensor([[0], [2], [7]])),
        components=[],
        properties=Labels(["p"], torch.tensor([[0], [1]])),
    )

    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.tensor(
                [[[14, 15], [10, 11]], [[12, 13], [14, 15]], [[10, 11], [12, 13]]]
            ),
            samples=Labels(
                ["sample", "g"],
                torch.tensor([[0, 1], [1, 1], [2, 1]]),
            ),
            components=[
                Labels(["c"], torch.tensor([[0], [1]])),
            ],
            properties=block_2.properties,
        ),
    )

    keys = Labels(names=["key_1", "key_2"], values=torch.tensor([[0, 0], [1, 0]]))
    return TensorMap(keys, [block_1, block_2])


@pytest.fixture
def tensor_sorted_descending():
    """
    This is the `tensor` fixture sorted in descending order how it should be returned
    when applying metatensor.operations.sort with `descending=True` option.
    """
    block_1 = TensorBlock(
        values=torch.tensor([[3, 5], [1, 2]]),
        samples=Labels(["s"], torch.tensor([[2], [0]])),
        components=[],
        properties=Labels(["p"], torch.tensor([[1], [0]])),
    )

    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.tensor([[[4, 9], [3, 8]], [[2, 7], [1, 6]]]),
            samples=Labels(
                ["sample", "g"],
                torch.tensor([[1, 1], [0, 1]]),
            ),
            components=[
                Labels(["c"], torch.tensor([[1], [0]])),
            ],
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=torch.tensor([[3, 4], [1, 2], [5, 6]]),
        samples=Labels(["s"], torch.tensor([[7], [2], [0]])),
        components=[],
        properties=Labels(["p"], torch.tensor([[1], [0]])),
    )
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.tensor(
                [[[11, 10], [15, 14]], [[15, 14], [13, 12]], [[13, 12], [11, 10]]]
            ),
            samples=Labels(
                ["sample", "g"],
                torch.tensor([[2, 1], [1, 1], [0, 1]]),
            ),
            components=[
                Labels(["c"], torch.tensor([[1], [0]])),
            ],
            properties=block_2.properties,
        ),
    )
    keys = Labels(
        names=["key_1", "key_2"],
        values=torch.tensor([[1, 0], [0, 0]]),
    )
    return TensorMap(keys, [block_2, block_1])


def test_sort_ascending(tensor, tensor_sorted_ascending):
    metatensor.torch.allclose_block_raise(
        metatensor.torch.sort_block(tensor.block(0)), tensor_sorted_ascending.block(1)
    )
    metatensor.torch.allclose_block_raise(
        metatensor.torch.sort_block(tensor.block(1)), tensor_sorted_ascending.block(0)
    )

    metatensor.torch.allclose_raise(
        metatensor.torch.sort(tensor), tensor_sorted_ascending
    )


def test_sort_descending(tensor, tensor_sorted_descending):
    metatensor.torch.allclose_block_raise(
        tensor_sorted_descending.block(0),
        metatensor.torch.sort_block(tensor.block(0), descending=True),
    )
    metatensor.torch.allclose_block_raise(
        tensor_sorted_descending.block(0),
        metatensor.torch.sort_block(tensor.block(0), descending=True),
    )


def test_high_numb():
    tensor = TensorMap(
        keys=Labels(
            names=["a", "b"],
            values=torch.tensor([[2, 1], [1, 0]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32),
                samples=Labels(
                    names=["s1", "s2", "s3"],
                    values=torch.tensor([[0, 1, 2], [2, 3, 4], [1, 5, 7]]),
                ),
                components=[],
                properties=Labels(
                    names=["p1", "p2"],
                    values=torch.tensor([[100, 0], [5, 7000]]),
                ),
            ),
            TensorBlock(
                values=torch.tensor(
                    [[2.2, 3.1, 4.1], [2.2, 1.1, 2.1], [2.2, 5.1, 6.1]]
                ),
                samples=Labels(
                    names=["s1", "s2", "s3"],
                    values=torch.tensor([[0, 2, 2], [0, 1, 2], [1, 5, 7]]),
                ),
                components=[],
                properties=Labels(
                    names=["p1", "p2"],
                    values=torch.tensor([[5, 10], [5, 5], [5, 6]]),
                ),
            ),
        ],
    )

    tensor_order = TensorMap(
        keys=Labels(
            names=["a", "b"],
            values=torch.tensor([[1, 0], [2, 1]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.tensor(
                    [[1.1, 2.1, 2.2], [3.1, 4.1, 2.2], [5.1, 6.1, 2.2]]
                ),
                samples=Labels(
                    names=["s1", "s2", "s3"],
                    values=torch.tensor([[0, 1, 2], [0, 2, 2], [1, 5, 7]]),
                ),
                components=[],
                properties=Labels(
                    names=["p1", "p2"],
                    values=torch.tensor([[5, 5], [5, 6], [5, 10]]),
                ),
            ),
            TensorBlock(
                values=torch.tensor([[2, 1], [6, 5], [4, 3]], dtype=torch.float32),
                samples=Labels(
                    names=["s1", "s2", "s3"],
                    values=torch.tensor([[0, 1, 2], [1, 5, 7], [2, 3, 4]]),
                ),
                components=[],
                properties=Labels(
                    names=["p1", "p2"],
                    values=torch.tensor([[5, 7000], [100, 0]]),
                ),
            ),
        ],
    )
    sorted = metatensor.torch.sort(tensor)
    metatensor.torch.allclose_raise(sorted, tensor_order)
