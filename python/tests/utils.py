from array import array
from unittest import result

import numpy as np

from equistore import Labels, TensorBlock, TensorMap


def test_tensor_map():
    """
    Create a dummy tensor map to be used in tests. This is the same one as the
    tensor map used in `tensor.rs` tests.
    """
    block_1 = TensorBlock(
        values=np.full((3, 1, 1), 1.0),
        samples=Labels(["samples"], np.array([[0], [2], [4]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_1.add_gradient(
        "parameter",
        samples=Labels(
            ["sample", "parameter"], np.array([[0, -2], [2, 3]], dtype=np.int32)
        ),
        data=np.full((2, 1, 1), 11.0),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
    )

    block_2 = TensorBlock(
        values=np.full((3, 1, 3), 2.0),
        samples=Labels(["samples"], np.array([[0], [1], [3]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[3], [4], [5]], dtype=np.int32)),
    )
    block_2.add_gradient(
        "parameter",
        data=np.full((3, 1, 3), 12.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, -2], [0, 3], [2, -2]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
    )

    block_3 = TensorBlock(
        values=np.full((4, 3, 1), 3.0),
        samples=Labels(["samples"], np.array([[0], [3], [6], [8]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_3.add_gradient(
        "parameter",
        data=np.full((1, 3, 1), 13.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[1, -2]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
    )

    block_4 = TensorBlock(
        values=np.full((4, 3, 1), 4.0),
        samples=Labels(["samples"], np.array([[0], [1], [2], [5]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_4.add_gradient(
        "parameter",
        data=np.full((2, 3, 1), 14.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, 1], [3, 3]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
    )

    # TODO: different number of components?

    keys = Labels(
        names=["key_1", "key_2"],
        values=np.array([[0, 0], [1, 0], [2, 2], [2, 3]], dtype=np.int32),
    )

    return TensorMap(keys, [block_1, block_2, block_3, block_4])


def test_large_tensor_map():
    """
    Create a dummy tensor map of 16 blocks to be used in tests. This is the same
    tensor map used in `tensor.rs` tests.
    """
    tensor = test_tensor_map()
    block_list = [block.copy() for _, block in tensor]

    for i in range(8):
        tmp_bl = TensorBlock(
            values=np.full((4, 3, 1), 4.0),
            samples=Labels(["samples"], np.array([[0], [1], [4], [5]], dtype=np.int32)),
            components=[
                Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))
            ],
            properties=Labels(["properties"], np.array([[i]], dtype=np.int32)),
        )
        tmp_bl.add_gradient(
            "parameter",
            data=np.full((2, 3, 1), 14.0),
            samples=Labels(
                ["sample", "parameter"],
                np.array([[0, 1], [3, 3]], dtype=np.int32),
            ),
            components=[
                Labels(
                    ["components"],
                    np.array([[0], [1], [2]], dtype=np.int32),
                )
            ],
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
            dtype=np.int32,
        ),
    )
    return TensorMap(keys, block_list)


def compare_blocks(block1: TensorBlock, block2: TensorBlock, rtol=1e-13):
    """
    Compare two ``TensorBlock``s to see if they are the same.
    This works only for numpy array.
    Returns a dictionary in which the keys are:
    global : bool -> are the two blocks globally equal ?
    values, properties, ... : bool -> one for each property of the block

    if the gradient is present a nested dictionary with the same structure is present in the returned dictionary.
    """
    result = {}
    result["values"] = np.allclose(block1.values, block2.values, rtol=rtol)
    result["samples"] = np.all(block1.samples == block2.samples)
    if len(block1.properties) == len(block2.properties):
        result["properties"] = np.all(
            [np.all(p1 == p2) for p1, p2 in zip(block1.properties, block2.properties)]
        )
    else:
        result["properties"] = False

    result["components"] = np.all(block1.components == block2.components)
    result["gradients"] = {}
    if block1.has_any_gradient() > 0 and len(block1.gradients_list()) == len(
        block2.gradients_list()
    ):
        result["gradients"]["general"] = True
        for parameter, gradient1 in block1.gradients():
            gradient2 = block2.gradient(parameter)
            result["gradients"][parameter] = {}
            result["gradients"][parameter]["samples"] = np.all(
                gradient1.samples == gradient2.samples
            )

            if len(gradient1.components) > 0 and len(gradient1.components) == len(
                gradient2.components
            ):
                result["gradients"][parameter]["components"] = np.all(
                    [
                        np.all(c1 == c2)
                        for c1, c2 in zip(gradient1.components, gradient2.components)
                    ]
                )
            elif len(gradient1.components) != len(gradient2.components):
                result["gradients"][parameter]["components"] = False
            else:
                result["gradients"][parameter]["components"] = True

            if len(gradient1.properties) > 0 and len(gradient1.properties) == len(
                gradient2.properties
            ):
                result["gradients"][parameter]["properties"] = np.all(
                    [
                        np.all(p1 == p2)
                        for p1, p2 in zip(gradient1.properties, gradient2.properties)
                    ]
                )
            elif len(gradient1.properties) != len(gradient2.properties):
                result["gradients"][parameter]["components"] = False
            else:
                result["gradients"][parameter]["properties"] = True

            result["gradients"][parameter]["data"] = np.allclose(
                gradient1.data, gradient2.data, rtol=1e-13
            )

            result["gradients"][parameter]["general"] = (
                result["gradients"][parameter]["samples"]
                and result["gradients"][parameter]["components"]
                and result["gradients"][parameter]["properties"]
                and result["gradients"][parameter]["data"]
            )
            result["gradients"]["general"] = (
                result["gradients"]["general"]
                and result["gradients"][parameter]["general"]
            )

    elif len(block1.gradients_list()) != len(block2.gradients_list()):
        result["gradients"]["general"] = False
    else:
        result["gradients"]["general"] = True

    result["general"] = (
        result["values"]
        and result["samples"]
        and result["properties"]
        and result["components"]
        and result["gradients"]["general"]
    )
    return result


def Xfun1(x, y, z):
    return np.arctan(-x + 2 * y * y + 3 * z * z * z)


def Xfun1_dx(x, y, z):
    """derivative w.r.t x of Xfun1"""
    return -1 / (1 + (-x + 2 * y * y + 3 * z * z * z) ** 2)


def Xfun1_dy(x, y, z):
    """derivative w.r.t y of Xfun1"""
    return 4 * y / (1 + (-x + 2 * y * y + 3 * z * z * z) ** 2)


def Xfun1_dz(x, y, z):
    """derivative w.r.t z of Xfun1"""
    return 9 * z * z / (1 + (-x + 2 * y * y + 3 * z * z * z) ** 2)


def Xfun2(x, y, z):
    return x**3 + 2 * y + 3 * z**2


def Xfun2_dx(x, y, z):
    """derivative w.r.t x of Xfun2"""
    return 3 * x**2


def Xfun2_dy(x, y, z):
    """derivative w.r.t y of Xfun2"""
    return 2


def Xfun2_dz(x, y, z):
    """derivative w.r.t z of Xfun2"""
    return 6 * z


def get_value_linear_solve():
    """Generate a value matrix for block and gradient in
    the test for the linear solve
    """
    data = np.arange(15).reshape((-1, 3))
    Xval = np.zeros((len(data), 2))
    Xgradval = np.zeros((len(data), 3, 2))
    for i in range(len(data)):
        Xval[i, 0] = Xfun1(data[i, 0], data[i, 1], data[i, 2])
        Xval[i, 1] = Xfun2(data[i, 0], data[i, 1], data[i, 2])
        Xgradval[i, 0, 0] = Xfun1_dx(data[i, 0], data[i, 1], data[i, 2])
        Xgradval[i, 1, 0] = Xfun1_dy(data[i, 0], data[i, 1], data[i, 2])
        Xgradval[i, 2, 0] = Xfun1_dz(data[i, 0], data[i, 1], data[i, 2])
        Xgradval[i, 0, 1] = Xfun2_dx(data[i, 0], data[i, 1], data[i, 2])
        Xgradval[i, 1, 1] = Xfun2_dy(data[i, 0], data[i, 1], data[i, 2])
        Xgradval[i, 2, 1] = Xfun2_dz(data[i, 0], data[i, 1], data[i, 2])

    w = np.array([[1], [3]])
    Yval = np.dot(Xval, w)
    Ygradval = np.dot(Xgradval, w)

    return Xval, Xgradval, Yval, Ygradval
