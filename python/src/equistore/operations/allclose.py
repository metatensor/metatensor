import numpy as np

from equistore import TensorBlock, TensorMap

from . import _dispatch
from ._utils import _check_same_gradients, _check_same_keys


def allclose(
    tensor1: TensorMap,
    tensor2: TensorMap,
    rtol=1e-13,
    atol=1e-12,
    equal_nan=False,
) -> bool:
    """Compare two :py:class:`TensorMap`.

    This function returns ``True`` if the two tensors have the same keys
    (potentially in different order) and all the :py:class:`TensorBlock` have
    the same samples, components, properties, and their values matrices pass the
    numpy-like ``allclose`` test with the provided ``rtol``, and ``atol``.

    The :py:class:`TensorMap` contains gradient data, then this function only
    returns ``True`` if all the gradients also have the same samples,
    components, properties and their data matrices pass the numpy-like
    ``allclose`` test with the provided ``rtol``, and ``atol``.

    In practice this function calls :py:func:`allclose_raise`, returning
    ``True`` if no exception is rased, `False` otherwise.

    :param tensor1: first :py:class:`TensorMap`.
    :param tensor2: second :py:class:`TensorMap`.
    :param rtol: relative tolerance for ``allclose``. Default: 1e-13.
    :param atol: absolute tolerance for ``allclose``. Defaults: 1e-12.
    :param equal_nan: should two ``NaN`` be considered equal? Defaults: False.
    """
    try:
        allclose_raise(
            tensor1=tensor1,
            tensor2=tensor2,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
        return True
    except ValueError:
        return False


def allclose_raise(
    tensor1: TensorMap,
    tensor2: TensorMap,
    rtol=1e-13,
    atol=1e-12,
    equal_nan=False,
):
    """
    Compare two :py:class:`TensorMap`, raising a ``ValueError`` if they are not
    the same.

    The message associated with the ``ValueError`` will contain more information
    on where the two :py:class:`TensorMap` differ. See :py:func:`allclose` for
    more information on which :py:class:`TensorMap` are considered equal.

    :param tensor1: first :py:class:`TensorMap`.
    :param tensor2: second :py:class:`TensorMap`.
    :param rtol: relative tolerance for ``allclose``. Default: 1e-13.
    :param atol: absolute tolerance for ``allclose``. Defaults: 1e-12.
    :param equal_nan: should two ``NaN`` be considered equal? Defaults: False.
    """
    _check_same_keys(tensor1, tensor2, "allclose")

    for key, block1 in tensor1:
        try:
            allclose_block_raise(
                block1,
                tensor2.block(key),
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )
        except ValueError as e:
            raise ValueError(f"The TensorBlocks with key = {key} are different") from e


def allclose_block(
    block1: TensorBlock,
    block2: TensorBlock,
    rtol=1e-13,
    atol=1e-12,
    equal_nan=False,
) -> bool:
    """
    Compare two :py:class:`TensorBlock`.

    This function returns ``True`` if the two :py:class:`TensorBlock` have the
    same samples, components, properties and their values matrices must pass the
    numpy-like ``allclose`` test with the provided ``rtol``, and ``atol``.

    If the :py:class:`TensorBlock` contains gradients, then the gradient must
    also have same samples, components, properties and their data matrices must
    pass the numpy-like ``allclose`` test with the provided ``rtol``, and
    ``atol``.

    In practice this function calls :py:func:`allclose_block_raise`, returning
    ``True`` if no exception is rased, `False` otherwise.

    :param block1: first :py:class:`TensorBlock`.
    :param block2: second :py:class:`TensorBlock`.
    :param rtol: relative tolerance for ``allclose``. Default: 1e-13.
    :param atol: absolute tolerance for ``allclose``. Defaults: 1e-12.
    :param equal_nan: should two ``NaN`` be considered equal? Defaults: False.
    """
    try:
        allclose_block_raise(
            block1=block1,
            block2=block2,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
        return True
    except ValueError:
        return False


def allclose_block_raise(
    block1: TensorBlock,
    block2: TensorBlock,
    rtol=1e-13,
    atol=1e-12,
    equal_nan=False,
):
    """
    Compare two :py:class:`TensorBlock`, raising a ``ValueError`` if they are
    not the same.

    The message associated with the ``ValueError`` will contain more information
    on where the two :py:class:`TensorBlock` differ. See
    :py:func:`allclose_block` for more information on which
    :py:class:`TensorBlock` are considered equal.

    :param block1: first :py:class:`TensorBlock`.
    :param block2: second :py:class:`TensorBlock`.
    :param rtol: relative tolerance for ``allclose``. Default: 1e-13.
    :param atol: absolute tolerance for ``allclose``. Defaults: 1e-12.
    :param equal_nan: should two ``NaN`` be considered equal? Defaults: False.
    """

    if not np.all(block1.values.shape == block2.values.shape):
        raise ValueError("values shapes are different")

    if not _dispatch.allclose(
        block1.values,
        block2.values,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    ):
        raise ValueError("values are not allclose")

    if not block1.samples.names == block2.samples.names:
        raise ValueError("samples names are not the same or not in the same order")

    if not np.all(block1.samples == block2.samples):
        raise ValueError("samples not the same or not in the same order")

    if not len(block1.properties) == len(block2.properties):
        raise ValueError("properties not the same or not in the same order")
    elif not block1.properties.names == block2.properties.names:
        raise ValueError("properties names are not the same or not in the same order")
    else:
        for p1, p2 in zip(block1.properties, block2.properties):
            if not np.all(p1 == p2):
                raise ValueError("properties not the same or not in the same order")

    if not (len(block1.components) == len(block2.components)):
        raise ValueError("different number of components")
    else:
        for c1, c2 in zip(block1.components, block2.components):
            if not (c1.names == c2.names):
                raise ValueError(
                    "components names are not the same or not in the same order"
                )

            if not np.all(c1 == c2):
                raise ValueError("components not the same or not in the same order")

    if len(block1.gradients_list()) > 0:
        _check_same_gradients(block1, block2, "allclose")

    for parameter, gradient1 in block1.gradients():
        gradient2 = block2.gradient(parameter)
        if not (gradient1.samples.names == gradient2.samples.names):
            raise ValueError(
                f'gradient ("{parameter}") samples names are not the same or '
                "not in the same order"
            )

        if len(gradient1.components) != len(gradient2.components):
            raise ValueError(f'different number of gradient ("{parameter}") components')

        for c1, c2 in zip(gradient1.components, gradient2.components):
            if not (c1.names == c2.names):
                raise ValueError(
                    f'gradient ("{parameter}") components names '
                    "are not the same or not in the same order"
                )

            if not np.all(c1 == c2):
                raise ValueError(
                    f'gradient ("{parameter}") components not the same or '
                    "not in the same order"
                )

        if len(gradient1.properties) != len(gradient2.properties):
            raise ValueError(f'different number of gradient ("{parameter}") properties')
        elif not (gradient1.properties.names == gradient2.properties.names):
            raise ValueError(
                f'gradient ("{parameter}") properties names are '
                "not the same or not in the same order"
            )
        else:
            for p1, p2 in zip(gradient1.properties, gradient2.properties):
                if not np.all(p1 == p2):
                    raise ValueError(
                        f'gradient ("{parameter}") properties not the same '
                        "or not in the same order"
                    )

        if not _dispatch.allclose(
            gradient1.data,
            gradient2.data,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        ):
            raise ValueError(f'gradient ("{parameter}") data are not allclose')
