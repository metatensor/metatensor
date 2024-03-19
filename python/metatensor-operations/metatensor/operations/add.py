from typing import List, Union

from ._backend import (
    TensorBlock,
    TensorMap,
    check_isinstance,
    torch_jit_is_scripting,
    torch_jit_script,
)
from ._utils import (
    _check_blocks_raise,
    _check_same_gradients_raise,
    _check_same_keys_raise,
)


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

        assert gradient_1.values.shape == gradient_2.values.shape
        assert gradient_1.samples == gradient_2.samples

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


@torch_jit_script
def add(A: TensorMap, B: Union[int, float, TensorMap]) -> TensorMap:
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

    if not torch_jit_is_scripting():
        if not check_isinstance(A, TensorMap):
            raise TypeError(f"`A` must be a metatensor TensorMap, not {type(A)}")

    blocks: List[TensorBlock] = []
    if torch_jit_is_scripting():
        is_tensor_map = isinstance(B, TensorMap)
    else:
        is_tensor_map = check_isinstance(B, TensorMap)

    if isinstance(B, (float, int)):
        B = float(B)
        for block_A in A.blocks():
            blocks.append(_add_block_constant(block=block_A, constant=B))

    elif is_tensor_map:
        _check_same_keys_raise(A, B, "add")
        for key, block_A in A.items():
            block_B = B[key]
            _check_blocks_raise(
                block_A,
                block_B,
                fname="add",
            )
            _check_same_gradients_raise(
                block_A,
                block_B,
                fname="add",
            )
            blocks.append(_add_block_block(block_1=block_A, block_2=block_B))
    else:
        if torch_jit_is_scripting():
            extra = ""
        else:
            extra = f", not {type(B)}"

        raise TypeError("`B` must be a metatensor TensorMap or a scalar value" + extra)

    return TensorMap(A.keys, blocks)
