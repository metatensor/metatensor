import numpy as np

from equistore import TensorBlock, TensorMap

from . import _dispatch
from .equal_metadata import _check_blocks, _check_same_gradients, _check_same_keys


def equal(tensor1: TensorMap, tensor2: TensorMap) -> bool:
    """Compare two :py:class:`TensorMap`.

    This function returns ``True`` if the two tensors have the same keys
    (potentially in different order) and all the :py:class:`TensorBlock` have
    the same (and in the same order) samples, components, properties,
    and their values matrices pass the == test with the provided
    ``rtol``, and ``atol``.

    The :py:class:`TensorMap` contains gradient data, then this function only
    returns ``True`` if all the gradients also have the same samples,
    components, properties and their data matrices pass == test.

    In practice this function calls :py:func:`equal_raise`, returning
    ``True`` if no exception is rased, `False` otherwise.

    :param tensor1: first :py:class:`TensorMap`.
    :param tensor2: second :py:class:`TensorMap`.
    """
    try:
        equal_raise(tensor1=tensor1, tensor2=tensor2)
        return True
    except ValueError:
        return False


def equal_raise(tensor1: TensorMap, tensor2: TensorMap):
    """
    Compare two :py:class:`TensorMap`, raising a ``ValueError`` if they are not
    the same.

    The message associated with the ``ValueError`` will contain more information
    on where the two :py:class:`TensorMap` differ. See :py:func:`equal` for
    more information on which :py:class:`TensorMap` are considered equal.

    :param tensor1: first :py:class:`TensorMap`.
    :param tensor2: second :py:class:`TensorMap`.
    """
    _check_same_keys(tensor1, tensor2, "equal")

    for key, block1 in tensor1:
        try:
            equal_block_raise(block1, tensor2.block(key))
        except ValueError as e:
            raise ValueError(f"The TensorBlocks with key = {key} are different") from e


def equal_block(block1: TensorBlock, block2: TensorBlock) -> bool:
    """
    Compare two :py:class:`TensorBlock`.

    This function returns ``True`` if the two :py:class:`TensorBlock` have the
    same samples, components, properties and their values matrices must pass == test.

    If the :py:class:`TensorBlock` contains gradients, then the gradient must
    also have same (and in the same order) samples, components, properties
    and their data matrices must pass the == test.

    In practice this function calls :py:func:`equal_block_raise`, returning
    ``True`` if no exception is rased, `False` otherwise.

    :param block1: first :py:class:`TensorBlock`.
    :param block2: second :py:class:`TensorBlock`.
    """
    try:
        equal_block_raise(block1=block1, block2=block2)
        return True
    except ValueError:
        return False


def equal_block_raise(block1: TensorBlock, block2: TensorBlock):
    """
    Compare two :py:class:`TensorBlock`, raising a ``ValueError`` if they are
    not the same.

    The message associated with the ``ValueError`` will contain more information
    on where the two :py:class:`TensorBlock` differ. See
    :py:func:`equal_block` for more information on which
    :py:class:`TensorBlock` are considered equal.

    :param block1: first :py:class:`TensorBlock`.
    :param block2: second :py:class:`TensorBlock`.
    """

    if not np.all(block1.values.shape == block2.values.shape):
        raise ValueError("values shapes are different")

    if not _dispatch.all(block1.values == block2.values):
        raise ValueError("values are not equal")
    _check_blocks(
        block1, block2, props=["samples", "properties", "components"], fname="equal"
    )
    _check_same_gradients(
        block1, block2, props=["samples", "properties", "components"], fname="equal"
    )

    for parameter, gradient1 in block1.gradients():
        gradient2 = block2.gradient(parameter)

        if not _dispatch.all(gradient1.data == gradient2.data):
            raise ValueError(f'gradient ("{parameter}") data are not equal')
