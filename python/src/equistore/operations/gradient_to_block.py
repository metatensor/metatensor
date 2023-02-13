from typing import List, Optional, Tuple

from equistore import TensorBlock, TensorMap


def gradient_to_block(
    tensor: TensorMap,
    remove: Optional[List[str]] = None,
) -> Tuple(TensorMap, dict):
    """
    Return a tuple containg a :py:class:`TensorMap` and a ``dictionary``.
    The first :py:class:`TensorMap` is the input ``tensor`` where you removed
    the selected gradients, the second is a dictionary with as keys the ``parameters``
    of the removed gradients and as values as :py:class:`TensorMap` with the same keys
    as ``tensor`` and :py:class:`TensorBlock` equal to the removed
    :py:class:`Gradients`.

    :param tensor: input :py:class:`TensorMap`, with gradients to remove
    :param remove: which gradients should be removed from ``tensor`` and included in
        the new :py:class:`TensorMap`.
        If this is set to ``None`` (this is the default), all the gradients will
        be removed.
    """

    if remove is None:
        remove = tensor.block(0).gradients_list()

    blocks = []
    gradient_blocks = {}
    for parameter in remove:
        gradient_blocks[parameter] = []
    for _, block in tensor:
        new_block = TensorBlock(
            values=block.values,
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        for parameter, gradient in block.gradients():
            if parameter in remove:
                _new_gradient_block = TensorBlock(
                    values=gradient.data,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=gradient.properties,
                )
                gradient_blocks[parameter].append(_new_gradient_block)

            new_block.add_gradient(
                parameter,
                gradient.data,
                gradient.samples,
                gradient.components,
            )

        blocks.append(new_block)

    gradient_map = {}
    for parameter in remove:
        gradient_map[parameter] = TensorMap(tensor.keys, gradient_blocks[parameter])
    return (TensorMap(tensor.keys, blocks), gradient_map)
