from typing import List

from . import _dispatch
from ._backend import TensorBlock, TensorMap, torch_jit_script


@torch_jit_script
def requires_grad_block(block: TensorBlock, requires_grad: bool = True) -> TensorBlock:
    """
    Set ``requires_grad`` on the values and all gradients in this ``block`` to the
    provided value.

    This is mainly intended for torch arrays, and will warn if trying to set
    ``requires_grad=True`` with numpy arrays.

    :param block: :py:class:`TensorBlock` to modify
    :param requires_grad: new value for ``requires_grad``
    """

    new_block = TensorBlock(
        values=_dispatch.requires_grad(block.values, value=requires_grad),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    for parameter, gradient in block.gradients():
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        new_block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=_dispatch.requires_grad(gradient.values, value=requires_grad),
                samples=gradient.samples,
                components=gradient.components,
                properties=gradient.properties,
            ),
        )

    return new_block


@torch_jit_script
def requires_grad(tensor: TensorMap, requires_grad: bool = True) -> TensorMap:
    """
    Set ``requires_grad`` on all arrays (blocks and gradients of blocks) in this
    ``tensor`` to the provided value.

    This is mainly intended for torch arrays, and will warn if trying to set
    ``requires_grad=True`` with numpy arrays.

    :param tensor: :py:class:`TensorMap` to modify
    :param requires_grad: new value for ``requires_grad``
    """

    blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        blocks.append(requires_grad_block(block, requires_grad=requires_grad))

    return TensorMap(tensor.keys, blocks)
