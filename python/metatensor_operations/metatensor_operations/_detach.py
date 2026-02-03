from typing import List

from . import _dispatch
from ._backend import TensorBlock, TensorMap, torch_jit_script


@torch_jit_script
def detach_block(block: TensorBlock) -> TensorBlock:
    """
    Detach all the values in this ``block`` and all of its gradient from any
    computational graph.

    This function is related but different to
    :py:func:`metatensor.remove_gradients_block`.
    :py:func:`metatensor.remove_gradients_block` can be used to remove the explicit
    forward mode gradients stored inside the ``block``, and this function detach the
    values (as well as any potential gradients) from the computational graph PyTorch
    uses for backward differentiation.
    """

    new_block = TensorBlock(
        values=_dispatch.detach(block.values),
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
                values=_dispatch.detach(gradient.values),
                samples=gradient.samples,
                components=gradient.components,
                properties=gradient.properties,
            ),
        )

    return new_block


@torch_jit_script
def detach(tensor: TensorMap) -> TensorMap:
    """
    Detach all the arrays in this ``tensor`` from any computational graph.

    This is useful for example when handling torch arrays, to be able to save them with
    :py:func:`metatensor.save` or :py:func:`metatensor.torch.save`.

    This function is related but different to :py:func:`metatensor.remove_gradients`.
    :py:func:`metatensor.remove_gradients` can be used to remove the explicit forward
    mode gradients stored inside the blocks, and this function detach the values (as
    well as any potential gradients) from the computational graph PyTorch uses for
    backward differentiation.
    """

    blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        blocks.append(detach_block(block))

    return TensorMap(tensor.keys, blocks)
