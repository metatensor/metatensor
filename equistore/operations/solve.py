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
    assert len(X.components) == 0, "solve for block with components is not implemented"
    assert len(Y.components) == 0, "solve for block with components is not implemented"

    if X.has_any_gradient():
        if len(X.gradients_list()) != len(Y.gradients_list()) or (
            not np.all(
                [parameter in Y.gradients_list() for parameter in X.gradients_list()]
            )
        ):
            raise ValueError("The two input TensorBlock should have the same gradients")
        for parameter, Xgradient in X.gradients():
            X_grad_data_reshape = Xgradient.data.reshape(-1, X.values.shape[-1])
            # TODO: this assume the atoms are in the same order in X_grad &
            # forces
            Ygradient = Y.gradient(parameter)
            Y_grad_data_reshape = Ygradient.data.reshape(-1, Y.values.shape[-1])
            valuesY = _dispatch.vstack((Y.values, Y_grad_data_reshape))
            valuesX = _dispatch.vstack((X.values, X_grad_data_reshape))
    else:
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
        values=valuesw.T,
        samples=Y.properties,
        components=[],
        properties=X.properties,
    )

    return w
