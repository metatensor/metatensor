from . import _dispatch
from ._backend import TensorBlock, TensorMap, torch_jit_script


@torch_jit_script
def is_contiguous_block(block: TensorBlock) -> bool:
    """
    Checks whether the values array and gradients values arrays (if present) of an input
    :py:class:`TensorBlock` are contiguous.

    Note that arrays of :py:class:`Labels` objects are not checked for contiguity.

    :param block: the input :py:class:`TensorBlock`.

    :return: bool, true if all values arrays contiguous, false otherwise.
    """
    if not _dispatch.is_contiguous(block.values):
        return False

    for _parameter, gradient in block.gradients():
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        if not _dispatch.is_contiguous(gradient.values):
            return False

    return True


@torch_jit_script
def is_contiguous(tensor: TensorMap) -> bool:
    """
    Checks whether all values arrays and gradients values arrays (if present) in all
    :py:class:`TensorBlock` of an input :py:class:`TensorMap` are contiguous.

    Note that arrays of :py:class:`Labels` objects are not checked for contiguity.

    :param tensor: the input :py:class:`TensorMap`.

    :return: bool, true if all values arrays contiguous, false otherwise.
    """
    for block in tensor.blocks():
        if not is_contiguous_block(block):
            return False

    return True
