import numpy as np

from equistore import TensorBlock, TensorMap

from . import _dispatch


def _dot_block(block1: TensorBlock, block2: TensorBlock) -> TensorBlock:
    """dot product among two `TensorBlocks`.
    result_block.values = block1.values @ block2.values.T
    The `TensorBlocks` should have the same `properties`
    """

    if not np.all(block1.properties == block2.properties):
        raise ValueError("the two TensorBlocks should have the same properties ")

    if len(block2.components) > 0:
        raise ValueError("The second TensorBlock should not have components ")

    if len(block2.gradients_list()) > 0:
        raise ValueError(
            "The second TensorBlock should not have gradient informations "
        )

    values1 = block1.values
    values2 = block2.values
    values = _dispatch.dot(values1, values2)

    result_block = TensorBlock(
        values=values,
        samples=block1.samples,
        components=block1.components,
        properties=block2.samples,
    )

    if len(block1.gradients_list()) > 0:
        for parameter in block1.gradients_list():
            gradient = block1.gradient(parameter)

            gradient_data = _dispatch.dot(
                gradient.data, values2
            )  # gradient.data @ block2.values.T

            result_block.add_gradient(
                parameter,
                gradient_data,
                gradient.samples,
                gradient.components,
            )

    return result_block


def dot(A: TensorMap, B: TensorMap) -> TensorMap:
    """
    Computes the dot product among two :py:class:`TensorMap`.
    The two :py:class:`TensorMap` must have the same ``keys``.
    The resulting :py:class:`TensorMap` will have the same keys of the two in input and
    each of its :py:class:`TensorBlock` will be the dot product
    of the two :py:class:`TensorBlock` of the input for the corresponding key.

    :py:class:`TensorBlocks` corresponding to the same key
    should have the same ``properties``
    The resulting :py:class:`TensorBlocks` of the dot product of
    two :py:class:`TensorBlocks` has \n
    :code:`result_block.values = block1.values @ block2.values.T`

    :param A: first :py:class:`TensorMap` to multiply
    :param B: second :py:class:`TensorMap` to multiply

    :return: a :py:class:`TensorMap` with the same keys of ``A`` and ``B``,
            and where each :py:class:`TensorBlock` has: the ``sample``
            equal to the ``sample`` of ``A``;
            the ``properties`` equal to the ``sample`` of ``B``;
            and the ``components`` equal to the
            ``components`` of ``A``
    """
    if len(A.keys) != len(B.keys) or (not np.all([key in B.keys for key in A.keys])):
        raise ValueError("The two input tensorMaps should have the same keys")
    blocks = []
    for key, block1 in A:
        block2 = B.block(key)
        blocks.append(_dot_block(block1=block1, block2=block2))
    return TensorMap(A.keys, blocks)
