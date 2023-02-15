import warnings

import numpy as np

from ..block import TensorBlock
from ..tensor import TensorMap
from . import _dispatch
from ._utils import _check_blocks, _check_maps


def lstsq(X: TensorMap, Y: TensorMap, rcond, driver=None) -> TensorMap:
    """
    Solve a linear system using two :py:class:`TensorMap`.

    Return the least-squares solution to a linear equation ``Y = X * w`` solving
    for ``w``, where ``Y``, ``X`` and ``w`` are all :py:class:`TensorMap`. ``Y``
    and ``X`` must have the same ``keys``.

    :param X: a :py:class:`TensorMap` containing the "coefficient" matrices.
    :param Y: a :py:class:`TensorMap` containing the "dependent variable" values.
    :param rcond: Cut-off ratio for small singular values of a. None chose the
                default value for numpy or PyTorch
    :param driver: Used only in torch (ignored if numpy is used), see
        https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html
        for a full description

    :return: a :py:class:`TensorMap` with the same keys of ``Y`` and ``X``, and
            where each :py:class:`TensorBlock` has: the ``sample`` equal to the
            ``properties`` of ``Y``; and the ``properties`` equal to the
            ``properties`` of ``X``.
    """
    if rcond is None:
        warnings.warn(
            "WARNING rcond is set to None, which will trigger the default "
            "behavior which is different between numpy and torch lstsq function, "
            "and might depend on the version you are using.",
            stacklevel=1,
        )

    _check_maps(X, Y, "lstsq")

    blocks = []
    for key, X_block in X:
        Y_block = Y.block(key)
        blocks.append(_lstsq_block(X_block, Y_block, rcond=rcond, driver=driver))

    return TensorMap(X.keys, blocks)


def _lstsq_block(X: TensorBlock, Y: TensorBlock, rcond, driver) -> TensorBlock:
    # TODO handle properties and samples not in the same order?

    if not np.all(X.samples == Y.samples):
        raise ValueError(
            "X and Y blocks in `lstsq` should have the same samples in the same order"
        )

    if len(X.components) > 0:
        if len(X.components) != len(Y.components):
            raise ValueError(
                "X and Y blocks in `lstsq` should have the same components \
                in the same order"
            )

        for x_component, y_component in zip(X.components, Y.components):
            if not np.all(x_component == y_component):
                raise ValueError(
                    "X and Y blocks in `lstsq` should have the same components \
                    in the same order"
                )

    # reshape components together with the samples
    X_n_properties = X.values.shape[-1]
    X_values = X.values.reshape(-1, X_n_properties)

    Y_n_properties = Y.values.shape[-1]
    Y_values = Y.values.reshape(-1, Y_n_properties)

    _check_blocks(X, Y, ["gradients"], "lstsq")

    for parameter, X_gradient in X.gradients():
        X_gradient_data = X_gradient.data.reshape(-1, X_n_properties)
        X_values = _dispatch.vstack((X_values, X_gradient_data))

        Y_gradient = Y.gradient(parameter)
        Y_gradient_data = Y_gradient.data.reshape(-1, Y_n_properties)
        Y_values = _dispatch.vstack((Y_values, Y_gradient_data))

    weights = _dispatch.lstsq(X_values, Y_values, rcond=rcond, driver=driver)

    return TensorBlock(
        values=weights.T,
        samples=Y.properties,
        components=[],
        properties=X.properties,
    )
