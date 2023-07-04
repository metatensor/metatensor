from typing import Union

import numpy as np

from equistore.core import TensorBlock, TensorMap

from . import _dispatch
from ._utils import _check_blocks, _check_same_gradients, _check_same_keys


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
        _check_same_keys(A, B, "divide")
        for key, block_A in A.items():
            block_B = B.block(key)
            _check_blocks(
                block_A,
                block_B,
                props=["samples", "components", "properties"],
                fname="divide",
            )
            _check_same_gradients(
                block_A,
                block_B,
                props=["samples", "components", "properties"],
                fname="divide",
            )
            blocks.append(_divide_block_block(block_1=block_A, block_2=block_B))

    elif isinstance(B, (float, int)):
        B = float(B)
        for block_A in A.blocks():
            blocks.append(_divide_block_constant(block=block_A, constant=B))

    else:
        raise TypeError("B should be a TensorMap or a scalar value")

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
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        result_block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=gradient.values / constant,
                samples=gradient.samples,
                components=gradient.components,
                properties=gradient.properties,
            ),
        )

    return result_block


def _divide_block_block(block_1: TensorBlock, block_2: TensorBlock) -> TensorBlock:
    values = block_1.values / block_2.values

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

        values_grad = []
        for i_sample in range(len(block_1.samples)):
            samples_1 = gradient_1.samples["sample"].values
            i_sample_grad_1 = np.where(samples_1 == i_sample)[0]

            samples_2 = gradient_2.samples["sample"].values
            i_sample_grad_2 = np.where(samples_2 == i_sample)[0]

            value_grad = (
                -block_1.values[i_sample]
                * gradient_2.values[i_sample_grad_2]
                / block_2.values[i_sample] ** 2
            )
            value_grad += gradient_1.values[i_sample_grad_1] / block_2.values[i_sample]
            values_grad.append(value_grad)

        values_grad = _dispatch.concatenate(values_grad, axis=0)

        result_block.add_gradient(
            parameter=parameter_1,
            gradient=TensorBlock(
                values=values_grad,
                samples=gradient_1.samples,
                components=gradient_1.components,
                properties=gradient_1.properties,
            ),
        )

    return result_block
