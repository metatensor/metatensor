from equistore.core import TensorBlock, TensorMap

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

    if isinstance(B, (float, int)):
        B = float(B)
    else:
        raise TypeError("B should be a scalar value")

    for block_A in A.blocks():
        blocks.append(_pow_block_constant(block=block_A, constant=B))

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
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        values_grad = []
        gradient_values = gradient.values
        # I want the difference between the number of components of the gradients and
        # the values
        diff_components = len(gradient_values.shape) - len(block.values.shape)
        gradient_samples_sample = gradient.samples["sample"].values[:, 0]
        values_grad.append(
            constant
            * gradient_values
            * block.values[gradient_samples_sample].reshape(
                (-1,) + (1,) * diff_components + _shape
            )
            ** (constant - 1)
        )
        values_grad = _dispatch.concatenate(values_grad, axis=0)

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
