from typing import Union

import numpy as np

from ..block import TensorBlock
from ..tensor import TensorMap
from . import _dispatch
from .equal_metadata import _check_blocks, _check_maps, _check_same_gradients


def divide(A: TensorMap, B: Union[float, TensorMap]) -> TensorMap:
    r"""Return a new :class:`TensorMap` with the values being the element-wise
    division of ``A`` and ``B``.

    If ``B`` is a :py:class:`TensorMap` it has to have the same metadata as ``A``.

    If gradients are present in ``A``:

    *  ``B`` is a scalar then:

       .. math::
            \nabla(A / B) =  \nabla A / B

    *  ``B`` is a :py:class:`TensorMap` with the same metadata of ``A``.
        The multiplication is performed with the rule of the derivatives:

       .. math::
            \nabla(A / B) =(B*\nabla A-A*\nabla B)/B^2

    :param A: First :py:class:`TensorMap` for the division.
    :param B: Second instance for the division. Parameter can be a scalar
            or a :py:class:`TensorMap`. In the latter case ``B`` must have the same
            metadata of ``A``.

    :return: New :py:class:`TensorMap` with the same metadata as ``A``.
    """

    blocks = []
    if isinstance(B, TensorMap):
        _check_maps(A, B, "divide")
        for key, blockA in A:
            blockB = B.block(key)
            _check_blocks(
                blockA,
                blockB,
                props=["samples", "components", "properties"],
                fname="divide",
            )
            _check_same_gradients(
                blockA,
                blockB,
                props=["samples", "components", "properties"],
                fname="divide",
            )
            blocks.append(_divide_block_block(block1=blockA, block2=blockB))
    else:
        # check if can be converted in float (so if it is a constant value)
        try:
            float(B)
        except TypeError as e:
            raise TypeError("B should be a TensorMap or a scalar value. ") from e
        for blockA in A.blocks():
            blocks.append(_divide_block_constant(block=blockA, constant=B))

    return TensorMap(A.keys, blocks)


def _divide_block_constant(block: TensorBlock, constant: float) -> TensorBlock:
    values = block.values / constant

    result_block = TensorBlock(
        values=values,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    for parameter, gradient in block.gradients():
        result_block.add_gradient(
            parameter,
            gradient.data / constant,
            gradient.samples,
            gradient.components,
        )

    return result_block


def _divide_block_block(block1: TensorBlock, block2: TensorBlock) -> TensorBlock:
    values = block1.values / block2.values

    result_block = TensorBlock(
        values=values,
        samples=block1.samples,
        components=block1.components,
        properties=block1.properties,
    )

    for parameter1, gradient1 in block1.gradients():
        gradient2 = block2.gradient(parameter1)
        values_grad = []
        for isample in range(len(block1.samples)):
            isample_grad1 = np.where(gradient1.samples["sample"] == isample)[0]
            isample_grad2 = np.where(gradient2.samples["sample"] == isample)[0]
            values_grad.append(
                -block1.values[isample]
                * gradient2.data[isample_grad2]
                / block2.values[isample] ** 2
                + gradient1.data[isample_grad1] / block2.values[isample]
            )
        values_grad = _dispatch.concatenate(values_grad, axis=0)

        result_block.add_gradient(
            parameter1,
            values_grad,
            gradient1.samples,
            gradient1.components,
        )

    return result_block
