import numpy as np

from equistore import TensorBlock, TensorMap

from . import _dispatch
from .equal_metadata import _check_blocks, _check_maps, _check_same_gradients


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
    the same (and in the same order) samples, components, properties,
    and their values matrices pass the numpy-like ``allclose`` test with the provided
    ``rtol``, and ``atol``.

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
    _check_maps(tensor1, tensor2, "allclose")

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
    also have same (and in the same order) samples, components, properties
    and their data matrices must pass the numpy-like ``allclose`` test with the
    provided ``rtol``, and ``atol``.

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
    _check_blocks(
        block1, block2, props=["samples", "properties", "components"], fname="allclose"
    )
    _check_same_gradients(
        block1, block2, props=["samples", "properties", "components"], fname="allclose"
    )

    for parameter, gradient1 in block1.gradients():
        gradient2 = block2.gradient(parameter)

        if not _dispatch.allclose(
            gradient1.data,
            gradient2.data,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        ):
            raise ValueError(f'gradient ("{parameter}") data are not allclose')
