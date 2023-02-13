import numpy as np

from ..block import TensorBlock
from ..tensor import TensorMap
from . import _dispatch


def pow(A: TensorMap, B: float) -> TensorMap:
    r"""Return a new :class:`TensorMap` with the same metadata of ``A``
    and the values being the element-wise ``B``-power of ``A.values``.

    ``B`` can only be a scalar.
    If gradients are present in ``A`` the gradient of the resulting :class:`TensorMap`
    are given by the standard formula:

    .. math::
        \nabla(A ^ B) = B* \nabla A * A^{(B-1)}

    :param A: :py:class:`TensorMap` to be elevated at the power of B.
    :param B: The power to which we want to elevate ``A``.
               Parameter can be a scalar

    :return: New :py:class:`TensorMap` with the same metadata as ``A``.
    """

    blocks = []

    # check if can be converted in float (so if it is a constant value)
    try:
        float(B)
    except TypeError as e:
        raise TypeError("B should be a scalar value. ") from e
    for blockA in A.blocks():
        blocks.append(_pow_block_constant(block=blockA, constant=B))

    return TensorMap(A.keys, blocks)


def _pow_block_constant(block: TensorBlock, constant: float) -> TensorBlock:
    values = block.values[:] ** constant

    result_block = TensorBlock(
        values=values,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    for parameter, gradient in block.gradients():
        values_grad = []
        for isample in range(len(block.samples)):
            isample_grad1 = np.where(gradient.samples["sample"] == isample)[0]
            values_grad.append(
                constant
                * gradient.data[isample_grad1]
                * block.values[isample] ** (constant - 1)
            )
        values_grad = _dispatch.vstack(values_grad)

        result_block.add_gradient(
            parameter,
            values_grad,
            gradient.samples,
            gradient.components,
        )

    return result_block
