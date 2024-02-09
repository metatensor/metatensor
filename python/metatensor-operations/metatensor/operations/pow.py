from typing import List, Union

from . import _dispatch
from ._backend import (
    TensorBlock,
    TensorMap,
    check_isinstance,
    torch_jit_is_scripting,
    torch_jit_script,
)


def _pow_block_constant(block: TensorBlock, constant: float) -> TensorBlock:
    values = block.values[:] ** constant

    result_block = TensorBlock(
        values=values,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    _shape: List[int] = []
    for c in block.components:
        _shape.append(len(c))
    _shape.append(len(block.properties))

    for parameter, gradient in block.gradients():
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        gradient_values = gradient.values
        # we find the difference between the number of components
        # of the gradients and the values and then use it to create
        # empty dimensions for broadcasting
        diff_components = len(gradient_values.shape) - len(block.values.shape)
        gradient_samples_to_values_samples = gradient.samples.column("sample")
        values_grad = (
            constant
            * gradient_values
            * block.values[
                _dispatch.to_index_array(gradient_samples_to_values_samples)
            ].reshape([-1] + [1] * diff_components + _shape)
            ** (constant - 1)
        )

        result_block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=values_grad,
                samples=gradient.samples,
                components=gradient.components,
                properties=gradient.properties,
            ),
        )

    return result_block


@torch_jit_script
def pow(A: TensorMap, B: Union[float, int]) -> TensorMap:
    r"""Return a new :class:`TensorMap` with the same metadata of ``A``
    and the values being the element-wise ``B``-power of ``A.values``.

    ``B`` can only be a scalar.
    If gradients are present in ``A`` the gradient of the resulting :class:`TensorMap`
    are given by the standard formula:

    .. math::
        \nabla(A ^ B) = B* \nabla A * A^{(B-1)}

    :param A: :py:class:`TensorMap` to be elevated at the power of B.
    :param B: The power to which we want to elevate ``A``.
               Parameter can only be a scalar or something that can be converted to a
               scalar.

    :return: New :py:class:`TensorMap` with the same metadata as ``A``.
    """
    if not torch_jit_is_scripting():
        if not check_isinstance(A, TensorMap):
            raise TypeError(f"`A` must be a metatensor TensorMap, not {type(A)}")

        if not isinstance(B, (float, int)):
            raise TypeError(f"`B` must be a scalar value, not {type(B)}")

    B = float(B)

    blocks: List[TensorBlock] = []
    for block_A in A.blocks():
        blocks.append(_pow_block_constant(block=block_A, constant=B))

    return TensorMap(A.keys, blocks)
