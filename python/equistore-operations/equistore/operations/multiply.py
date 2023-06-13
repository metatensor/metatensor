from typing import Union

import numpy as np

from equistore.core import TensorBlock, TensorMap

from . import _dispatch
from ._utils import _check_blocks, _check_same_gradients, _check_same_keys


def multiply(A: TensorMap, B: Union[float, TensorMap]) -> TensorMap:
    r"""Return a new :class:`TensorMap` with the values being the element-wise
    multiplication of ``A`` and ``B``.

    If ``B`` is a :py:class:`TensorMap` it has to have the same metadata as ``A``.

    If gradients are present in ``A``:

    *  ``B`` is a scalar then:

       .. math::
            \nabla(A * B) = B * \nabla A

    *  ``B`` is a :py:class:`TensorMap` with the same metadata of ``A``.
        The multiplication is performed with the rule of the derivatives:

       .. math::
            \nabla(A * B) = B * \nabla A + A * \nabla B

    :param A: First :py:class:`TensorMap` for the multiplication.
    :param B: Second instance for the multiplication. Parameter can be a scalar
            or a :py:class:`TensorMap`. In the latter case ``B`` must have the same
            metadata of ``A``.

    :return: New :py:class:`TensorMap` with the same metadata as ``A``.
    """
    blocks = []
    if isinstance(B, TensorMap):
        _check_same_keys(A, B, "multiply")
        for key in A.keys:
            blockA = A[key]
            blockB = B[key]
            _check_blocks(
                blockA,
                blockB,
                props=["samples", "components", "properties"],
                fname="multiply",
            )
            _check_same_gradients(
                blockA,
                blockB,
                props=["samples", "components", "properties"],
                fname="multiply",
            )
            blocks.append(_multiply_block_block(block_1=blockA, block_2=blockB))
    else:
        # check if can be converted in float (so if it is a constant value)
        try:
            float(B)
        except TypeError as e:
            raise TypeError("B should be a TensorMap or a scalar value. ") from e
        for blockA in A.blocks():
            blocks.append(_multiply_block_constant(block=blockA, constant=B))

    return TensorMap(A.keys, blocks)


def _multiply_block_constant(block: TensorBlock, constant: float) -> TensorBlock:
    values = constant * block.values

    result_block = TensorBlock(
        values=values,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    for parameter, gradient in block.gradients():
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        result_block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=gradient.values * constant,
                samples=gradient.samples,
                components=gradient.components,
                properties=gradient.properties,
            ),
        )

    return result_block


def _multiply_block_block(block_1: TensorBlock, block_2: TensorBlock) -> TensorBlock:
    values = block_1.values * block_2.values

    result_block = TensorBlock(
        values=values,
        samples=block_1.samples,
        components=block_1.components,
        properties=block_1.properties,
    )

    for parameter_1, gradient_1 in block_1.gradients():
        gradient_2 = block_2.gradient(parameter_1)

        if len(gradient_1.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        if len(gradient_2.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        gradient_values = []
        for i_sample in range(len(block_1.samples)):
            gradient_samples_1 = gradient_1.samples["sample"].values
            i_sample_grad_1 = np.where(gradient_samples_1 == i_sample)[0]

            gradient_samples_2 = gradient_2.samples["sample"].values
            i_sample_grad_2 = np.where(gradient_samples_2 == i_sample)[0]

            gradient_values.append(
                block_1.values[i_sample] * gradient_2.values[i_sample_grad_2]
                + gradient_1.values[i_sample_grad_1] * block_2.values[i_sample]
            )
        gradient_values = _dispatch.concatenate(gradient_values, axis=0)

        result_block.add_gradient(
            parameter=parameter_1,
            gradient=TensorBlock(
                values=gradient_values,
                samples=gradient_1.samples,
                components=gradient_1.components,
                properties=gradient_1.properties,
            ),
        )

    return result_block
