import warnings

import numpy as np

from ..labels import Labels
from ..block import TensorBlock
from ..tensor import TensorMap
from . import _dispatch
from .equal_metadata import _check_maps, _check_same_gradients


def lstsq(X: TensorMap, Y: TensorMap, rcond, driver=None) -> TensorMap:
    """
    Solve a linear system using two :py:class:`TensorMap`.

    Return the least-squares solution to a linear equation ``Y = X * w`` solving
    for ``w``, where ``Y``, ``X`` and ``w`` are all :py:class:`TensorMap`. ``Y``
    and ``X`` must have the same ``keys``.

    .. note::
      The values of ``w`` differ from the output of numpy or torch in that they
      are already transposed. Be aware of that if you want to manually access
      values of ``w`` (see also the example below).

    Here is an example using this function:

    >>> values_X = np.array([
    ...     [1.0, 2.0],
    ...     [3.0, 1.0],
    ... ])
    ...
    >>> values_Y = np.array([
    ...     [1.0, 0.0],
    ...     [0.0, 1.0],
    ... ])
    ...
    >>> samples = Labels(
    ...     ["structure"],
    ...     np.array([[0], [1]]),
    ... )
    ...
    >>> components = []
    ...
    >>> properties = Labels(
    ...     ["properties"],
    ...     np.array([[0], [1]])
    ... )
    ...
    >>> keys = Labels(names=["key"], values=np.array([[0]]))
    ...
    >>> block_X = TensorBlock(
    ...     values_X, samples, components, properties,
    ... )
    ...
    >>> block_Y = TensorBlock(
    ...     values_Y, samples, components, properties,
    ... )
    ...
    >>> X = TensorMap(keys, [block_X])
    ...
    >>> Y = TensorMap(keys, [block_Y])
    ...
    >>> w = lstsq(X, Y, rcond=1e-10)
    ...
    >>> # Note: we take the transpose here
    >>> print(X.block(0).values @ w.block(0).values.T)
    [[ 1.00000000e+00 -2.22044605e-16]
     [-3.33066907e-16  1.00000000e+00]]


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

    _check_same_gradients(X, Y, props=None, fname="lstsq")

    for parameter, X_gradient in X.gradients():
        X_gradient_data = X_gradient.data.reshape(-1, X_n_properties)
        X_values = _dispatch.concatenate((X_values, X_gradient_data), axis=0)

        Y_gradient = Y.gradient(parameter)
        Y_gradient_data = Y_gradient.data.reshape(-1, Y_n_properties)
        Y_values = _dispatch.concatenate((Y_values, Y_gradient_data), axis=0)

    weights = _dispatch.lstsq(X_values, Y_values, rcond=rcond, driver=driver)

    return TensorBlock(
        values=weights.T,
        samples=Y.properties,
        components=[],
        properties=X.properties,
    )
