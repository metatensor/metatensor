from typing import List, Optional

from ._classes import TensorBlock, TensorMap


def remove_gradients(
    tensor: TensorMap,
    remove: Optional[List[str]] = None,
) -> TensorMap:
    """Remove some or all of the gradients from a :py:class:`TensorMap`.

    :param tensor:
        input :py:class:`TensorMap`, with gradients to remove

    :param remove:
        which gradients should be excluded from the new tensor map. If this is
        set to :py:obj:`None` (this is the default), all the gradients will be
        removed.

    :returns:
        A new tensormap without the gradients.
    """

    if remove is None:
        remove = tensor.block(0).gradients_list()

    blocks: List[TensorBlock] = []
    for block in tensor.blocks():
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

        blocks.append(new_block)

    return TensorMap(tensor.keys, blocks)
