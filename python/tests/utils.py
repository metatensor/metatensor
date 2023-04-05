"""Utility functions to be used accross all test files.

These functions are by design no pytest fixtures to avoid a confusing global import.
"""

import numpy as np

from equistore import Labels, TensorBlock, TensorMap


def tensor():
    """
    Create a dummy tensor map to be used in tests. This is the same one as the
    tensor map used in `tensor.rs` tests.
    """
    block_1 = TensorBlock(
        values=np.full((3, 1, 1), 1.0),
        samples=Labels(["samples"], np.array([[0], [2], [4]])),
        components=[Labels(["components"], np.array([[0]]))],
        properties=Labels(["properties"], np.array([[0]])),
    )
    block_1.add_gradient(
        "parameter",
<<<<<<< HEAD
        samples=Labels(["sample", "parameter"], np.array([[0, -2], [2, 3]])),
        data=np.full((2, 1, 1), 11.0),
        components=[Labels(["components"], np.array([[0]]))],
=======
        samples=Labels(
            ["sample", "parameter"], np.array([[0, -2], [2, 3]], dtype=np.int32)
        ),
        values=np.full((2, 1, 1), 11.0),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
>>>>>>> 2bf62b5 (Change gradient.data to gradient.values everywhere)
    )

    block_2 = TensorBlock(
        values=np.full((3, 1, 3), 2.0),
        samples=Labels(["samples"], np.array([[0], [1], [3]])),
        components=[Labels(["components"], np.array([[0]]))],
        properties=Labels(["properties"], np.array([[3], [4], [5]])),
    )
    block_2.add_gradient(
        "parameter",
        values=np.full((3, 1, 3), 12.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, -2], [0, 3], [2, -2]]),
        ),
        components=[Labels(["components"], np.array([[0]]))],
    )

    block_3 = TensorBlock(
        values=np.full((4, 3, 1), 3.0),
        samples=Labels(["samples"], np.array([[0], [3], [6], [8]])),
        components=[Labels.arange("components", 3)],
        properties=Labels(["properties"], np.array([[0]])),
    )
    block_3.add_gradient(
        "parameter",
        values=np.full((1, 3, 1), 13.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[1, -2]]),
        ),
        components=[Labels.arange("components", 3)],
    )

    block_4 = TensorBlock(
        values=np.full((4, 3, 1), 4.0),
        samples=Labels(["samples"], np.array([[0], [1], [2], [5]])),
        components=[Labels.arange("components", 3)],
        properties=Labels(["properties"], np.array([[0]])),
    )
    block_4.add_gradient(
        "parameter",
        values=np.full((2, 3, 1), 14.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, 1], [3, 3]]),
        ),
        components=[Labels.arange("components", 3)],
    )

    # TODO: different number of components?

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
    block_list = [block.copy() for _, block in tensor()]

    for i in range(8):
        tmp_bl = TensorBlock(
            values=np.full((4, 3, 1), 4.0),
            samples=Labels(["samples"], np.array([[0], [1], [4], [5]])),
            components=[Labels.arange("components", 3)],
            properties=Labels(["properties"], np.array([[i]])),
        )
        tmp_bl.add_gradient(
            "parameter",
            values=np.full((2, 3, 1), 14.0),
            samples=Labels(
                ["sample", "parameter"],
                np.array([[0, 1], [3, 3]]),
            ),
            components=[Labels.arange("components", 3)],
        )
        block_list.append(tmp_bl)

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
    return TensorMap(keys, block_list)
