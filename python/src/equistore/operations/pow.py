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
               Parameter can only be a scalar or something that can be converted to a
               scalar.

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

    _shape = ()
    for c in block.components:
        _shape += (len(c),)
    _shape += (len(block.properties),)

    for parameter, gradient in block.gradients():
        values_grad = []
        gradient_data = gradient.data
        # I want the difference between the number of components of the gradients and
        # the values
        diff_components = len(gradient_data.shape) - len(block.values.shape)
        values_grad.append(
            constant
            * gradient_data
            * block.values[gradient.samples["sample"]].reshape(
                (-1,) + (1,) * diff_components + _shape
            )
            ** (constant - 1)
        )
        values_grad = _dispatch.vstack(values_grad)

        result_block.add_gradient(
            parameter,
            values_grad,
            gradient.samples,
            gradient.components,
        )

    return result_block
