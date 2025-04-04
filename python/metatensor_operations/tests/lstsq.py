import os

import numpy as np
import pytest

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_self_lstsq_no_gradients():
    block_1 = TensorBlock(
        values=np.array([[1, 2], [3, 5]]),
        samples=Labels("s", np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_2 = TensorBlock(
        values=np.array([[1, 2], [3, 4], [5, 6]]),
        samples=Labels("s", np.array([[0], [2], [7]])),
        components=[],
        properties=Labels.range("p", 2),
    )

    block_3 = TensorBlock(
        values=np.array([[1], [2]]),
        samples=Labels("s", np.array([[0], [2]])),
        components=[],
        properties=Labels("p", np.array([[0]])),
    )
    block_4 = TensorBlock(
        values=np.array([[23], [53], [83]]),
        samples=Labels("s", np.array([[0], [2], [7]])),
        components=[],
        properties=Labels("p", np.array([[6]])),
    )
    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
    X = TensorMap(keys, [block_1, block_2])
    Y = TensorMap(keys, [block_3, block_4])

    # Trying to do this with solve -> Raise Error

    message = "the values in each block of X should be a square 2D array"
    with pytest.raises(ValueError, match=message):
        w = metatensor.solve(X, Y)

    # solve with least square
    w = metatensor.lstsq(X, Y, rcond=1e-13)

    assert w.keys == X.keys
    assert np.allclose(w.block(0).values, np.array([-1.0, 1.0]), rtol=1e-13)
    assert np.allclose(w.block(1).values, np.array([7, 8]), rtol=1e-7)
    for key, block_w in w.items():
        assert block_w.samples == Y.block(key).properties
        assert block_w.properties == X.block(key).properties

    Ydot = metatensor.dot(X, w)
    assert metatensor.allclose(Ydot, Y)


def test_self_lstsq_gradients():
    x, x_grad, y, y_grad = get_value_linear_solve()
    block_X = TensorBlock(
        values=x,
        samples=Labels.range("s", 5),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_X.add_gradient(
        parameter="z",
        gradient=TensorBlock(
            values=x_grad,
            samples=Labels.range("sample", 5),
            components=[Labels.range("c", 3)],
            properties=block_X.properties,
        ),
    )

    block_Y = TensorBlock(
        values=y,
        samples=Labels.range("s", 5),
        components=[],
        properties=Labels("p", np.array([[2]])),
    )
    block_Y.add_gradient(
        parameter="z",
        gradient=TensorBlock(
            values=y_grad,
            samples=Labels.range("sample", 5),
            components=[Labels.range("c", 3)],
            properties=block_Y.properties,
        ),
    )

    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0]]))

    X = TensorMap(keys, [block_X])
    Y = TensorMap(keys, [block_Y])
    w = metatensor.lstsq(X, Y, rcond=1e-13)

    assert w.keys == X.keys
    assert np.allclose(w.block(0).values, np.array([1.0, 3.0]), rtol=1e-13)

    for key, block_w in w.items():
        assert block_w.samples == Y.block(key).properties
        assert block_w.properties == X.block(key).properties

    Ydot = metatensor.dot(X, w)
    assert metatensor.allclose(Ydot, Y)


def test_self_lstsq_gradients_components():
    x, x_grad, y, y_grad = get_value_linear_solve()
    block_X = TensorBlock(
        values=x.reshape((1, x.shape[0], x.shape[1])),
        samples=Labels("s", np.array([[0]])),
        components=[Labels.range("c", 5)],
        properties=Labels.range("p", 2),
    )
    block_X.add_gradient(
        parameter="z",
        gradient=TensorBlock(
            values=x_grad.reshape(1, 3, len(x), x.shape[-1]),
            samples=Labels.range("sample", 1),
            components=[
                Labels.range("der_components", 3),
                Labels.range("c", 5),
            ],
            properties=block_X.properties,
        ),
    )

    block_Y = TensorBlock(
        values=y.reshape((1, len(y), y.shape[-1])),
        samples=Labels("s", np.array([[0]])),
        components=[Labels.range("c", 5)],
        properties=Labels("p", np.array([[2]])),
    )
    block_Y.add_gradient(
        parameter="z",
        gradient=TensorBlock(
            values=y_grad.reshape((1, 3, len(y), y.shape[-1])),
            samples=Labels.range("sample", 1),
            components=[
                Labels.range("der_components", 3),
                Labels.range("c", 5),
            ],
            properties=block_Y.properties,
        ),
    )

    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0]]))

    X = TensorMap(keys, [block_X])
    Y = TensorMap(keys, [block_Y])
    w = metatensor.lstsq(X, Y, rcond=1e-13)

    assert w.keys == X.keys
    assert np.allclose(w.block(0).values, np.array([1.0, 3.0]), rtol=1e-13)

    for key, block_w in w.items():
        assert block_w.samples == Y.block(key).properties
        assert block_w.properties == X.block(key).properties

    Ydot = metatensor.dot(X, w)
    assert metatensor.allclose(Ydot, Y)


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
    """
    Generate a value matrix for block and gradient in the test for the linear
    solve
    """
    data = np.arange(15).reshape((-1, 3))
    x = np.zeros((len(data), 2))
    x_grad = np.zeros((len(data), 3, 2))
    for i in range(len(data)):
        x[i, 0] = Xfun1(data[i, 0], data[i, 1], data[i, 2])
        x[i, 1] = Xfun2(data[i, 0], data[i, 1], data[i, 2])
        x_grad[i, 0, 0] = Xfun1_dx(data[i, 0], data[i, 1], data[i, 2])
        x_grad[i, 1, 0] = Xfun1_dy(data[i, 0], data[i, 1], data[i, 2])
        x_grad[i, 2, 0] = Xfun1_dz(data[i, 0], data[i, 1], data[i, 2])
        x_grad[i, 0, 1] = Xfun2_dx(data[i, 0], data[i, 1], data[i, 2])
        x_grad[i, 1, 1] = Xfun2_dy(data[i, 0], data[i, 1], data[i, 2])
        x_grad[i, 2, 1] = Xfun2_dz(data[i, 0], data[i, 1], data[i, 2])

    w = np.array([[1], [3]])
    y = np.dot(x, w)
    y_grad = np.dot(x_grad, w)

    return x, x_grad, y, y_grad
