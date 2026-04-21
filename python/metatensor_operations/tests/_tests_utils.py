"""Utility functions to be used across all test files.

These functions are by design no pytest fixtures to avoid a confusing global import.
"""

import numpy as np

from metatensor import Labels, TensorBlock, TensorMap


def tensor():
    """A dummy tensor map to be used in tests"""
    block_1 = TensorBlock(
        values=np.full((3, 1, 1), 1.0),
        samples=Labels(["s"], np.array([[0], [2], [4]])),
        components=[Labels(["c"], np.array([[0]]))],
        properties=Labels(["p"], np.array([[0]])),
    )
    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            samples=Labels(["sample", "g"], np.array([[0, -2], [2, 3]])),
            values=np.full((2, 1, 1), 11.0),
            components=block_1.components,
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=np.full((3, 1, 3), 2.0),
        samples=Labels(["s"], np.array([[0], [1], [3]])),
        components=[Labels(["c"], np.array([[0]]))],
        properties=Labels(["p"], np.array([[3], [4], [5]])),
    )
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.full((3, 1, 3), 12.0),
            samples=Labels(["sample", "g"], np.array([[0, -2], [0, 3], [2, -2]])),
            components=block_2.components,
            properties=block_2.properties,
        ),
    )

    block_3 = TensorBlock(
        values=np.full((4, 3, 1), 3.0),
        samples=Labels(["s"], np.array([[0], [3], [6], [8]])),
        components=[Labels.range("c", 3)],
        properties=Labels(["p"], np.array([[0]])),
    )
    block_3.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.full((1, 3, 1), 13.0),
            samples=Labels(["sample", "g"], np.array([[1, -2]])),
            components=block_3.components,
            properties=block_3.properties,
        ),
    )

    block_4 = TensorBlock(
        values=np.full((4, 3, 1), 4.0),
        samples=Labels(["s"], np.array([[0], [1], [2], [5]])),
        components=[Labels.range("c", 3)],
        properties=Labels(["p"], np.array([[0]])),
    )
    block_4.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.full((2, 3, 1), 14.0),
            samples=Labels(["sample", "g"], np.array([[0, 1], [3, 3]])),
            components=block_4.components,
            properties=block_4.properties,
        ),
    )

    keys = Labels(
        names=["key_1", "key_2"],
        values=np.array([[0, 0], [1, 0], [2, 2], [2, 3]]),
    )

    return TensorMap(keys, [block_1, block_2, block_3, block_4])


def large_tensor():
    """
    Create a dummy tensor map of 16 blocks to be used in tests. This is the same
    tensor map used in `tensor.rs` tests.
    """
    blocks = [block.copy() for block in tensor()]

    for i in range(8):
        block = TensorBlock(
            values=np.full((4, 3, 1), 4.0),
            samples=Labels(["s"], np.array([[0], [1], [4], [5]])),
            components=[Labels.range("c", 3)],
            properties=Labels(["p"], np.array([[i]])),
        )
        block.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.full((2, 3, 1), 14.0),
                samples=Labels(["sample", "g"], np.array([[0, 1], [3, 3]])),
                components=block.components,
                properties=block.properties,
            ),
        )
        blocks.append(block)

    keys = Labels(
        names=["key_1", "key_2"],
        values=np.array(
            [
                [0, 0],
                [1, 0],
                [2, 2],
                [2, 3],
                [0, 4],
                [1, 4],
                [2, 4],
                [3, 4],
                [0, 5],
                [1, 5],
                [2, 5],
                [3, 5],
            ],
        ),
    )
    return TensorMap(keys, blocks)


# ---------------------------------------------------------------------------
# Shared helpers for info propagation tests
# ---------------------------------------------------------------------------

_INFO = {"description": "test_value", "key2": "value2"}


def tensor_with_info() -> TensorMap:
    """Return the standard test tensor with two info entries set."""
    t = tensor()
    t.set_info("description", "test_value")
    t.set_info("key2", "value2")
    return t


def simple_tensor() -> TensorMap:
    """Return a minimal one-block TensorMap with info set."""
    block = TensorBlock(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        samples=Labels(["s"], np.array([[0], [1]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )
    t = TensorMap(Labels(["key"], np.array([[0]])), [block])
    t.set_info("description", "test_value")
    return t


def multi_sample_tensor() -> TensorMap:
    """Tensor with 'system'+'atom' sample axes for reduction tests."""
    block = TensorBlock(
        values=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        samples=Labels(
            ["system", "atom"],
            np.array([[0, 0], [0, 1], [1, 0]]),
        ),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )
    t = TensorMap(Labels(["key"], np.array([[0]])), [block])
    t.set_info("description", "test_value")
    return t


def check_info(result: TensorMap, expected: dict):
    """Assert that result.info() contains exactly the expected key/value pairs."""
    info = result.info()
    for key, val in expected.items():
        assert key in info, f"info key '{key}' missing from result"
        assert info[key] == val, f"info['{key}'] = {info[key]!r}, expected {val!r}"
