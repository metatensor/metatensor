from typing import Union

from ..block import TensorBlock
from ..tensor import TensorMap
from ._utils import _check_blocks, _check_same_gradients_components, _check_same_keys


def add(A: TensorMap, B: Union[float, TensorMap]) -> TensorMap:
    """Return a new :class:`TensorMap` with the values being the sum of ``A`` and ``B``.

    If ``B`` is a :py:class:`TensorMap` it has to have the same metadata as ``A``.

    If gradients are present in ``A`` a sum is only performed if ``B`` is
    a :py:class:`TensorMap` as well.

    :param A: First :py:class:`TensorMap` for the addition.
    :param B: Second instance for the addition. Parameter can be a scalar or a
              :py:class:`TensorMap`. In the latter case ``B`` must have the same
              metadata of ``A``.

    :return: New :py:class:`TensorMap` with the same metadata as ``A``.
    """

    blocks = []
    if isinstance(B, TensorMap):
        _check_same_keys(A, B, "add")
        for key, blockA in A:
            blockB = B.block(key)
            _check_blocks(
                blockA,
                blockB,
                props=["samples", "components", "properties", "gradients"],
                fname="add",
            )
            _check_same_gradients_components(blockA, blockB, "add")
            blocks.append(_add_block_block(block1=blockA, block2=blockB))
    else:
        # check if can be converted in float (so if it is a constant value)
        try:
            float(B)
        except TypeError as e:
            raise TypeError("B should be a TensorMap or a scalar value. ") from e
        for blockA in A.blocks():
            blocks.append(_add_block_constant(block=blockA, constant=B))

    return TensorMap(A.keys, blocks)


def _add_block_constant(block: TensorBlock, constant: float) -> TensorBlock:
    values = constant + block.values

    result_block = TensorBlock(
        values=values,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    for parameter, gradient in block.gradients():
        result_block.add_gradient(
            parameter,
            gradient.data,
            gradient.samples,
            gradient.components,
        )

    return result_block


def _add_block_block(block1: TensorBlock, block2: TensorBlock) -> TensorBlock:
    # values = block1.values @ block2.values.T
    values = block1.values + block2.values

    result_block = TensorBlock(
        values=values,
        samples=block1.samples,
        components=block1.components,
        properties=block1.properties,
    )

    for parameter1, gradient1 in block1.gradients():
        result_block.add_gradient(
            parameter1,
            gradient1.data + block2.gradient(parameter1).data,
            gradient1.samples,
            gradient1.components,
        )

    return result_block
