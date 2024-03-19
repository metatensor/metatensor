from typing import List, Optional

from ._backend import TensorBlock, TensorMap, torch_jit_script


@torch_jit_script
def remove_gradients_block(
    block: TensorBlock,
    remove: Optional[List[str]] = None,
) -> TensorBlock:
    """Remove some or all of the gradients from a :py:class:`TensorBlock`.

    This function is related but different to :py:func:`metatensor.detach_block`. This
    function removes the explicit forward mode gradients stored in the ``block``, while
    :py:func:`metatensor.detach_block` separate the values (as well as any potential
    gradients) from the underlying computational graph use by PyTorch to run backward
    differentiation.

    :param block: :py:class:`TensorBlock` with gradients to be removed

    :param remove: which gradients should be excluded from the new block. If this is set
        to :py:obj:`None` (this is the default), all the gradients will be removed.

    :returns: A new :py:class:`TensorBlock` without the gradients in ``remove``.
    """

    if remove is None:
        remove = block.gradients_list()

    new_block = TensorBlock(
        values=block.values,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    for parameter, gradient in block.gradients():
        if parameter in remove:
            continue

        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        new_block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=gradient.values,
                samples=gradient.samples,
                components=gradient.components,
                properties=gradient.properties,
            ),
        )

    return new_block


@torch_jit_script
def remove_gradients(
    tensor: TensorMap,
    remove: Optional[List[str]] = None,
) -> TensorMap:
    """Remove some or all of the gradients from a :py:class:`TensorMap`.

    This function is related but different to :py:func:`metatensor.detach`. This
    function removes the explicit forward mode gradients stored in the blocks, while
    :py:func:`metatensor.detach` separate the values (as well as any potential
    gradients) from the underlying computational graph use by PyTorch to run backward
    differentiation.

    :param tensor: :py:class:`TensorMap` with gradients to be removed

    :param remove: which gradients should be excluded from the new tensor map. If this
        is set to :py:obj:`None` (this is the default), all the gradients will be
        removed.

    :returns: A new :py:class:`TensorMap` without the gradients in ``remove``.
    """

    if remove is None and len(tensor) != 0:
        remove = tensor.block(0).gradients_list()

    blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        blocks.append(remove_gradients_block(block, remove))

    return TensorMap(tensor.keys, blocks)
