from . import _dispatch
from ._backend import TensorBlock, TensorMap, torch_jit_script
from ._utils import (
    NotEqualError,
    _check_blocks_impl,
    _check_same_gradients_impl,
    _check_same_keys_impl,
)


def _equal_impl(tensor_1: TensorMap, tensor_2: TensorMap) -> str:
    """Abstract function to perform an equal operation between two TensorMaps."""
    message = _check_same_keys_impl(tensor_1, tensor_2, "equal")
    if message != "":
        return f"the tensor maps have different keys: {message}"

    for key, block_1 in tensor_1.items():
        message = _equal_block_impl(block_1=block_1, block_2=tensor_2.block(key))
        if message != "":
            return f"blocks for key {key.print()} are different: {message}"
    return ""


def _equal_block_impl(block_1: TensorBlock, block_2: TensorBlock) -> str:
    """Abstract function to perform an equal operation between two TensorBlocks."""
    if not block_1.values.shape == block_2.values.shape:
        return "values shapes are different"

    if not _dispatch.all(block_1.values == block_2.values):
        return "values are not equal"

    check_blocks_message = _check_blocks_impl(block_1, block_2, fname="equal")
    if check_blocks_message != "":
        return check_blocks_message

    check_same_gradient_message = _check_same_gradients_impl(
        block_1, block_2, fname="equal"
    )
    if check_same_gradient_message != "":
        return check_same_gradient_message

    for parameter, gradient1 in block_1.gradients():
        gradient2 = block_2.gradient(parameter)

        if not _dispatch.all(gradient1.values == gradient2.values):
            return f"gradient '{parameter}' values are not equal"
    return ""


@torch_jit_script
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
    return not bool(_equal_impl(tensor_1=tensor_1, tensor_2=tensor_2))


@torch_jit_script
def equal_raise(tensor_1: TensorMap, tensor_2: TensorMap) -> None:
    """
    Compare two :py:class:`TensorMap`, raising :py:class:`NotEqualError` if they
    are not the same.

    The message associated with the exception will contain more information on
    where the two :py:class:`TensorMap` differ. See :py:func:`equal` for more
    information on which :py:class:`TensorMap` are considered equal.

    :raises: :py:class:`metatensor.NotEqualError` if the blocks are
        different

    :param tensor_1: first :py:class:`TensorMap`.
    :param tensor_2: second :py:class:`TensorMap`.
    """
    message = _equal_impl(tensor_1=tensor_1, tensor_2=tensor_2)
    if message != "":
        raise NotEqualError(message)


@torch_jit_script
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
    return not bool(_equal_block_impl(block_1=block_1, block_2=block_2))


@torch_jit_script
def equal_block_raise(block_1: TensorBlock, block_2: TensorBlock) -> None:
    """
    Compare two :py:class:`TensorBlock`, raising
    :py:class:`metatensor.NotEqualError` if they are not the same.

    The message associated with the exception will contain more information on
    where the two :py:class:`TensorBlock` differ. See :py:func:`equal_block` for
    more information on which :py:class:`TensorBlock` are considered equal.

    :raises: :py:class:`metatensor.NotEqualError` if the blocks are different

    :param block_1: first :py:class:`TensorBlock`.
    :param block_2: second :py:class:`TensorBlock`.
    """
    message = _equal_block_impl(block_1=block_1, block_2=block_2)
    if message != "":
        raise NotEqualError(message)
