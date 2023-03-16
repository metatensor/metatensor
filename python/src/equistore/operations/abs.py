"""
Module to find the absolute values of a :py:class:`TensorMap`, returning a new
:py:class:`TensorMap`.
"""
from ..block import TensorBlock
from ..tensor import TensorMap
from . import _dispatch


def abs(A: TensorMap) -> TensorMap:
    """
    Return a new :py:class:`TensorMap` with the same metadata as A and absolute
    values of `A`.

    ..math::
    
        \nabla(|A|) = (A/|A|)*\nabla A

    :param A: input :py:class:`TensorMap` whose absolute values are needed.

    :return: New :py:class:`TensorMap` with the same metadata as ``A`` and
        absolute values of ``A``.
    """
    blocks = []
    keys = A.keys
    for key in keys:
        blocks.append(_abs_block(block=A[key]))
    return TensorMap(keys, blocks)


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
    _shape = ()
    for c in block.components:
        _shape += (len(c),)
    _shape += (len(block.properties),)

    for parameter, gradient in block.gradients():
        diff_components = len(gradient.components) - len(block.components)
        # The sign_values have the same dimensions as that of the block.values.
        # Reshape the sign_values to allow multiplication with gradient.data
        new_grad = gradient.data[:] * sign_values[gradient.samples["sample"]].reshape(
            (-1,) + (1,) * diff_components + _shape
        )

        result_block.add_gradient(
            parameter,
            new_grad,
            gradient.samples,
            gradient.components,
        )

    return result_block
