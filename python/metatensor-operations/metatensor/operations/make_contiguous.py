from typing import List

from . import _dispatch
from ._backend import TensorBlock, TensorMap, torch_jit_script


@torch_jit_script
def make_contiguous_block(block: TensorBlock) -> TensorBlock:
    """
    Returns a new :py:class:`TensorBlock` where the values and gradient (if present)
    arrays are made to be contiguous.

    :param block: the input :py:class:`TensorBlock`.
    """
    new_block = TensorBlock(
        values=_dispatch.make_contiguous(block.values),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    for parameter, gradient in block.gradients():
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        new_gradient = TensorBlock(
            values=_dispatch.make_contiguous(gradient.values),
            samples=gradient.samples,
            components=gradient.components,
            properties=gradient.properties,
        )
        new_block.add_gradient(parameter, new_gradient)

    return new_block


@torch_jit_script
def make_contiguous(tensor: TensorMap) -> TensorMap:
    """
    Returns a new :py:class:`TensorMap` where all values and gradient values arrays are
    made to be contiguous.

    :param tensor: the input :py:class:`TensorMap`.
    """
    new_blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        new_blocks.append(make_contiguous_block(block))

    return TensorMap(tensor.keys, new_blocks)
