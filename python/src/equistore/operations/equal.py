import numpy as np

from equistore import TensorBlock, TensorMap

from . import _dispatch
from .equal_metadata import _check_blocks, _check_same_gradients, _check_same_keys


class NotEqualError(Exception):
    """Exception used to indicate that two equistore objects are different"""

    pass


def equal(tensor_1: TensorMap, tensor_2: TensorMap) -> bool:
    """Compare two :py:class:`TensorMap`.

    This function returns :py:obj:`True` if the two tensors have the same keys
    (potentially in different order) and all the :py:class:`TensorBlock` have
    the same (and in the same order) samples, components, properties, and their
    their values are strictly equal.

    The :py:class:`TensorMap` contains gradient data, then this function only
    returns :py:obj:`True` if all the gradients also have the same samples,
    components, properties and their their values are strictly equal.

    In practice this function calls :py:func:`equal_raise`, returning
    :py:obj:`True` if no exception is raised, :py:obj:`False` otherwise.

    :param tensor_1: first :py:class:`TensorMap`.
    :param tensor_2: second :py:class:`TensorMap`.
    """
    try:
        equal_raise(tensor_1=tensor_1, tensor_2=tensor_2)
        return True
    except NotEqualError:
        return False


def equal_raise(tensor_1: TensorMap, tensor_2: TensorMap):
    """
    Compare two :py:class:`TensorMap`, raising :py:class:`NotEqualError` if they
    are not the same.

    The message associated with the exception will contain more information on
    where the two :py:class:`TensorMap` differ. See :py:func:`equal` for more
    information on which :py:class:`TensorMap` are considered equal.

    :raises: :py:class:`equistore.NotEqualError` if the blocks are
        different

    :param tensor_1: first :py:class:`TensorMap`.
    :param tensor_2: second :py:class:`TensorMap`.
    """
    try:
        _check_same_keys(tensor_1, tensor_2, "equal")
    except ValueError as e:
        raise NotEqualError("the tensor maps have different keys") from e

    for key, block_1 in tensor_1:
        try:
            equal_block_raise(block_1, tensor_2.block(key))
        except NotEqualError as e:
            raise NotEqualError(f"blocks for key '{key}' are different") from e


def equal_block(block_1: TensorBlock, block_2: TensorBlock) -> bool:
    """
    Compare two :py:class:`TensorBlock`.

    This function returns :py:obj:`True` if the two :py:class:`TensorBlock` have
    the same samples, components, properties and their values are strictly
    equal.

    If the :py:class:`TensorBlock` contains gradients, then the gradient must
    also have same (and in the same order) samples, components, properties and
    their values are strictly equal.

    In practice this function calls :py:func:`equal_block_raise`, returning
    :py:obj:`True` if no exception is raised, :py:obj:`False` otherwise.

    :param block_1: first :py:class:`TensorBlock`.
    :param block_2: second :py:class:`TensorBlock`.
    """
    try:
        equal_block_raise(block_1=block_1, block_2=block_2)
        return True
    except NotEqualError:
        return False


def equal_block_raise(block_1: TensorBlock, block_2: TensorBlock):
    """
    Compare two :py:class:`TensorBlock`, raising
    :py:class:`equistore.NotEqualError` if they are not the same.

    The message associated with the exception will contain more information on
    where the two :py:class:`TensorBlock` differ. See :py:func:`equal_block` for
    more information on which :py:class:`TensorBlock` are considered equal.

    :raises: :py:class:`equistore.NotEqualError` if the blocks are different

    :param block_1: first :py:class:`TensorBlock`.
    :param block_2: second :py:class:`TensorBlock`.
    """

    if not np.all(block_1.values.shape == block_2.values.shape):
        raise NotEqualError("values shapes are different")

    if not _dispatch.all(block_1.values == block_2.values):
        raise NotEqualError("values are not equal")

    try:
        _check_blocks(
            block_1,
            block_2,
            props=["samples", "properties", "components"],
            fname="equal",
        )
    except ValueError as e:
        raise NotEqualError(str(e))

    try:
        _check_same_gradients(
            block_1,
            block_2,
            props=["samples", "properties", "components"],
            fname="equal",
        )
    except ValueError as e:
        raise NotEqualError(str(e))

    for parameter, gradient1 in block_1.gradients():
        gradient2 = block_2.gradient(parameter)

        if not _dispatch.all(gradient1.data == gradient2.data):
            raise NotEqualError(f"gradient '{parameter}' values are not equal")
