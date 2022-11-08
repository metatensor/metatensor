from typing import List, Optional

from equistore import TensorBlock, TensorMap


def remove_gradients(
    tensor: TensorMap,
    remove: Optional[List[str]] = None,
) -> TensorMap:
    """
    Create a new :py:class:`TensorMap` without some or
    all of the gradients from ``tensor``.

    :param tensor: input :py:class:`TensorMap`, with gradients to remove
    :param remove: which gradients should be excluded from the new tensor map.
        If this is set to ``None`` (this is the default), all the gradients will
        be removed.
    """

    if remove is None:
        remove = tensor.block(0).gradients_list()

    blocks = []
    for _, block in tensor:
        new_block = TensorBlock(
            values=block.values,
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        for parameter, gradient in block.gradients():
            if parameter in remove:
                continue

            new_block.add_gradient(
                parameter,
                gradient.data,
                gradient.samples,
                gradient.components,
            )

        blocks.append(new_block)

    return TensorMap(tensor.keys, blocks)
