from typing import Union

from ._classes import TensorBlock, TensorMap
from ._utils import (
    _check_blocks_raise,
    _check_same_gradients_raise,
    _check_same_keys_raise,
)


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
        _check_same_keys_raise(A, B, "add")
        for key, block_A in A.items():
            block_B = B[key]
            _check_blocks_raise(
                block_A,
                block_B,
                check=("samples", "components", "properties"),
                fname="add",
            )
            _check_same_gradients_raise(
                block_A,
                block_B,
                check=("samples", "components", "properties"),
                fname="add",
            )
            blocks.append(_add_block_block(block_1=block_A, block_2=block_B))

    elif isinstance(B, (float, int)):
        B = float(B)
        for block_A in A.blocks():
            blocks.append(_add_block_constant(block=block_A, constant=B))

    else:
        raise TypeError("B should be a TensorMap or a scalar value")

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
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        result_block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=gradient.values,
                samples=gradient.samples,
                components=gradient.components,
                properties=gradient.properties,
            ),
        )

    return result_block


def _add_block_block(block_1: TensorBlock, block_2: TensorBlock) -> TensorBlock:
    values = block_1.values + block_2.values

    result_block = TensorBlock(
        values=values,
        samples=block_1.samples,
        components=block_1.components,
        properties=block_1.properties,
    )

    for parameter, gradient_1 in block_1.gradients():
        gradient_2 = block_2.gradient(parameter)

        if len(gradient_1.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        if len(gradient_2.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        result_block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=gradient_1.values + gradient_2.values,
                samples=gradient_1.samples,
                components=gradient_1.components,
                properties=gradient_1.properties,
            ),
        )

    return result_block
