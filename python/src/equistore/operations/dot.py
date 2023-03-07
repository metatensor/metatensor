import numpy as np

from ..block import TensorBlock
from ..tensor import TensorMap
from . import _dispatch
from ._utils import _check_maps


def dot(A: TensorMap, B: TensorMap) -> TensorMap:
    """Compute the dot product of two :py:class:`TensorMap`.

    The two :py:class:`TensorMap` must have the same ``keys``. The resulting
    :py:class:`TensorMap` will have the same keys as the input and each block
    will be the dot product of the two corresponding :py:class:`TensorBlock` in
    the input.

    :py:class:`TensorBlocks` corresponding to the same key must have the same
    ``properties``. The resulting :py:class:`TensorBlocks` of the dot product of
    two :py:class:`TensorBlocks` has ``result_block.values = block1.values @
    block2.values.T``

    :param A: first :py:class:`TensorMap` to multiply
    :param B: second :py:class:`TensorMap` to multiply

    :return: a :py:class:`TensorMap` with the same keys of ``A`` and ``B``, and
            where each :py:class:`TensorBlock` has: the ``sample`` equal to the
            ``sample`` of ``A``; the ``properties`` equal to the ``sample`` of
            ``B``; and the ``components`` equal to the ``components`` of ``A``
    """
    _check_maps(A, B, "dot")

    blocks = []
    for key, block1 in A:
        block2 = B.block(key)
        blocks.append(_dot_block(block1=block1, block2=block2))

    return TensorMap(A.keys, blocks)


def _dot_block(block1: TensorBlock, block2: TensorBlock) -> TensorBlock:
    if not np.all(block1.properties == block2.properties):
        raise ValueError("TensorBlocks in `dot` should have the same properties")

    if len(block2.components) > 0:
        raise ValueError("the second TensorMap in `dot` should not have components")

    if len(block2.gradients_list()) > 0:
        raise ValueError("the second TensorMap in `dot` should not have gradients")

    # values = block1.values @ block2.values.T
    values = _dispatch.dot(block1.values, block2.values)

    result_block = TensorBlock(
        values=values,
        samples=block1.samples,
        components=block1.components,
        properties=block2.samples,
    )

    for parameter, gradient in block1.gradients():
        # gradient_data = gradient.data @ block2.values.T
        gradient_data = _dispatch.dot(gradient.data, block2.values)

        result_block.add_gradient(
            parameter,
            gradient_data,
            gradient.samples,
            gradient.components,
        )

    return result_block
