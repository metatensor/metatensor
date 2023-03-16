from typing import Union

from ..block import TensorBlock
from ..tensor import TensorMap
from ._utils import _check_blocks, _check_maps, _check_same_gradients


def add(A: TensorMap, B: Union[float, TensorMap]) -> TensorMap:
    r"""Return a new :class:`TensorMap` with the values being the sum of
    ``A`` and ``B``.

    If ``B`` is a :py:class:`TensorMap` it has to have the same metadata as ``A``.

    If gradients are present in ``A``:

    *  ``B`` is a scalar:

       .. math::
            \nabla(A + B) = \nabla A

    * ``B`` is a :py:class:`TensorMap` with the same metadata of ``A``:

       .. math::
            \nabla(A + B) = \nabla A + \nabla B

    :param A: First :py:class:`TensorMap` for the addition.
    :param B: Second instance for the addition. Parameter can be a scalar or a
              :py:class:`TensorMap`. In the latter case ``B`` must have the same
              metadata of ``A``.

    :return: New :py:class:`TensorMap` with the same metadata as ``A``.
    """

    blocks = []
    if isinstance(B, TensorMap):
        _check_maps(A, B, "add")
        for key in A.keys:
            blockA = A[key]
            blockB = B[key]
            _check_blocks(
                blockA,
                blockB,
                props=["samples", "components", "properties"],
                fname="add",
            )
            _check_same_gradients(
                blockA,
                blockB,
                props=["samples", "components", "properties"],
                fname="add",
            )
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
            parameter=parameter,
            data=gradient.data,
            samples=gradient.samples,
            components=gradient.components,
        )

    return result_block


def _add_block_block(block1: TensorBlock, block2: TensorBlock) -> TensorBlock:
    values = block1.values + block2.values

    result_block = TensorBlock(
        values=values,
        samples=block1.samples,
        components=block1.components,
        properties=block1.properties,
    )

    for parameter1, gradient1 in block1.gradients():
        result_block.add_gradient(
            parameter=parameter1,
            data=gradient1.data + block2.gradient(parameter1).data,
            samples=gradient1.samples,
            components=gradient1.components,
        )

    return result_block
