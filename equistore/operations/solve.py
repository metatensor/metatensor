import numpy as np

from equistore import TensorBlock, TensorMap

from . import _dispatch


def solve(X: TensorMap, Y: TensorMap) -> TensorMap:
    """
    Solve a linear system among two :py:class:`TensorMap`.
    Solve the linear equation set X * w = Y for the unknown w.
    Where X , w, Y are all :py:class:`TensorMap`
    """

    if len(X) != len(Y) or (not np.all([key in Y.keys for key in X.keys])):
        raise ValueError("The two input TensorMap should have the same keys")

    blocks = []
    for key, blockX in X:
        blockY = Y.block(key)
        blocks.append(_solve_block(blockX, blockY))

    return TensorMap(X.keys, blocks)


def _solve_block(X: TensorBlock, Y: TensorBlock) -> TensorBlock:
    """
    Solve a linear system among two :py:class:`TensorBlock`.
    Solve the linear equation set X * w = Y for the unknown w.
    Where X , w, Y are all :py:class:`TensorBlock`
    """
    # TODO properties and samples not in the same order
    assert np.all(X.samples == Y.samples)
    valuesX = X.values
    valuesY = Y.values
    Xshape = valuesX.shape
    if len(Xshape) != 2:
        raise ValueError("X.values should be a 2D array")

    if Xshape[0] == Xshape[1]:
        # if X.values is square use solve
        valuesw = _dispatch.solve(valuesX, valuesY)
    else:
        # else use the lstsq
        valuesw = _dispatch.lstsq(valuesX, valuesY)

    w = TensorBlock(
        values=valuesw, samples=X.properties, components=[], properties=Y.properties
    )

    return w
