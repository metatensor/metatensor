"""
Module to find the absolute values of a :py:class:`TensorMap`, returning a new
:py:class:`TensorMap`.
"""

from typing import List

from . import _dispatch
from ._backend import TensorBlock, TensorMap, torch_jit_script


def _abs_block(block: TensorBlock) -> TensorBlock:
    """
    Returns a :py:class:`TensorBlock` with the absolute values of the block values and
    associated gradient data.
    """
    values = _dispatch.abs(block.values)

    result_block = TensorBlock(
        values=values,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    if len(block.gradients_list()) == 0:
        return result_block

    sign_values = _dispatch.sign(block.values)
    _shape: List[int] = []
    for c in block.components:
        _shape += [len(c)]
    _shape += [len(block.properties)]

    for parameter, gradient in block.gradients():
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        diff_components = len(gradient.components) - len(block.components)
        # The sign_values have the same dimensions as that of the block.values.
        # Reshape the sign_values to allow multiplication with gradient.values
        new_grad = gradient.values[:] * sign_values[
            _dispatch.to_index_array(gradient.samples.column("sample"))
        ].reshape([-1] + [1] * diff_components + _shape)

        gradient = TensorBlock(
            new_grad, gradient.samples, gradient.components, gradient.properties
        )
        result_block.add_gradient(parameter, gradient)

    return result_block


@torch_jit_script
def abs(A: TensorMap) -> TensorMap:
    r"""
    Return a new :py:class:`TensorMap` with the same metadata as A and absolute
    values of ``A``.

    .. math::
        A \rightarrow = \vert A \vert

    If gradients are present in ``A``:

    .. math::
        \nabla(A) \rightarrow \nabla(\vert A \vert) = (A/\vert A \vert)*\nabla A

    :param A: the input :py:class:`TensorMap`.

    :return: a new :py:class:`TensorMap` with the same metadata as ``A`` and
        absolute values of ``A``.
    """
    blocks: List[TensorBlock] = []
    keys = A.keys
    for i in range(len(keys)):
        blocks.append(_abs_block(block=A.block(keys.entry(i))))
    return TensorMap(keys, blocks)
