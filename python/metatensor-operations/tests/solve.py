import os

import numpy as np

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_self_solve_no_gradients():
    block_1 = TensorBlock(
        values=np.array([[1, 2], [3, 5]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_2 = TensorBlock(
        values=np.array([[1, 2], [3, 5]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_3 = TensorBlock(
        values=np.array([[1], [2]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0]])),
    )
    block_4 = TensorBlock(
        values=np.array([[2], [4]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0]])),
    )
    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
    X = TensorMap(keys, [block_1, block_2])
    Y = TensorMap(keys, [block_3, block_4])
    w = metatensor.solve(X, Y)

    assert w.keys == X.keys
    assert np.allclose(w.block(0).values, np.array([-1.0, 1.0]), rtol=1e-13)
    assert np.allclose(w.block(1).values, np.array([-2.0, 2.0]), rtol=1e-7)
    for key, block_w in w.items():
        assert block_w.samples == Y.block(key).properties
        assert block_w.properties == X.block(key).properties

    Ydot = metatensor.dot(X, w)
    assert metatensor.allclose(Ydot, Y)
