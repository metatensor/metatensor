import numpy as np

import metatensor as mts
from metatensor import Labels, TensorBlock, TensorMap
from metatensor.operations import _dispatch


def check_finite_differences(
    function,
    array,
    *,
    parameter: str,
    displacement: float = 1e-6,
    rtol: float = 1e-3,
    atol: float = 1e-9,
) -> None:
    """
    Check that analytical gradients with respect to ``parameter`` in the
    :py:class:`TensorMap` returned by ``function`` agree with finite differences.

    The ``function`` must take an array (either torch or numpy) and return a
    :py:class:`TensorMap`. All the blocks in the returned TensorMap should have one
    sample per row of the ``array``, and the gradient-specific components must match the
    other dimensions of the ``array``.
    """
    n_samples = array.shape[0]
    n_grad_components = array.shape[1:]

    reference = function(array)

    values_components = reference.block(0).components
    grad_components = reference.block(0).gradient(parameter).components

    assert len(grad_components) == len(values_components) + len(n_grad_components)

    for sample_i in range(n_samples):
        for grad_components_i in np.ndindex(n_grad_components):
            array_pos = _dispatch.copy(array)
            index = (sample_i,) + grad_components_i
            array_pos[index] += displacement / 2
            updated_pos = function(array_pos)

            array_neg = _dispatch.copy(array)
            array_neg[index] -= displacement / 2
            updated_neg = function(array_neg)

            assert updated_pos.keys == reference.keys
            assert updated_neg.keys == reference.keys

            for key, block in reference.items():
                gradients = block.gradient(parameter)

                block_pos = updated_pos.block(key)
                block_neg = updated_neg.block(key)

                for gradient_i, gradient_sample in enumerate(gradients.samples):
                    current_sample_i = gradient_sample[0]
                    if current_sample_i != sample_i:
                        continue

                    assert block_pos.samples[sample_i] == block.samples[sample_i]
                    assert block_neg.samples[sample_i] == block.samples[sample_i]

                    value_pos = block_pos.values[sample_i]
                    value_neg = block_neg.values[sample_i]

                    grad_index = (gradient_i,) + grad_components_i
                    gradient = gradients.values[grad_index]

                    assert value_pos.shape == gradient.shape
                    assert value_neg.shape == gradient.shape

                    fdiff = (value_pos - value_neg) / displacement
                    if not np.allclose(fdiff, gradient, rtol=rtol, atol=atol):
                        components_str = ", ".join(
                            c[i].print()[1:-1]
                            for i, c in zip(
                                grad_components_i, gradients.components, strict=True
                            )
                        )
                        raise AssertionError(
                            f"Wrong gradient of {block.samples[sample_i].print()} "
                            f"(gradient samples {gradient_sample.print()}, "
                            f"components ({components_str})), \n"
                            f"with relative tolerance {rtol} and absolute tolerance "
                            f"{atol}:\n"
                            f"    DIRECT CALCULATION: {gradient}\n"
                            f"    FINITE DIFFERENCES: {fdiff}\n"
                        )


def tensor_with_grad_a(array, parameter) -> TensorMap:
    """
    Creates a TensorMap from a set of cartesian vectors according to the function:

    .. math::

        f(x, y, z) = x^3 + y^3 + z^3

        \\nabla f = (3x^2, 3y^2, 3z^2)

    The gradients are stored with the given ``parameter`` name.
    """
    n_samples = array.shape[0]
    assert array.shape == (n_samples, 3)

    values = _dispatch.sum(array**3, axis=1)
    values_grad = 3 * array**2

    block = mts.block_from_array(values.reshape(n_samples, 1), sample_names=["s"])
    block.add_gradient(
        parameter=parameter,
        gradient=TensorBlock(
            values=values_grad.reshape(n_samples, 3, 1),
            samples=Labels.range("sample", len(values)),
            components=[Labels.range("xyz", 3)],
            properties=block.properties,
        ),
    )

    return TensorMap(Labels.range("_", 1), [block])


def tensor_with_grad_b(array, parameter) -> TensorMap:
    """
    Creates a TensorMap from a set of cartesian vectors according to the function:

    .. math::

        f(x, y, z) = 3x + 2y + 8*z + 4

        \\nabla f = (3, 2, 8)

    The gradients are stored with the given ``parameter`` name.
    """
    n_samples = array.shape[0]
    assert array.shape == (n_samples, 3)

    values = 3 * array[:, 0] + 2 * array[:, 1] + 8 * array[:, 2] + 4

    values_grad = _dispatch.zeros_like(array, (n_samples, 3, 1))
    values_grad[:, 0] = 3 * _dispatch.ones_like(array, (n_samples, 1))
    values_grad[:, 1] = 2 * _dispatch.ones_like(array, (n_samples, 1))
    values_grad[:, 2] = 8 * _dispatch.ones_like(array, (n_samples, 1))

    block = mts.block_from_array(values.reshape(-1, 1), sample_names=["s"])
    block.add_gradient(
        parameter=parameter,
        gradient=TensorBlock(
            values=values_grad,
            samples=Labels.range("sample", len(values)),
            components=[Labels.range("xyz", 3)],
            properties=block.properties,
        ),
    )

    return TensorMap(Labels.range("_", 1), [block])


def test_basic_functions():
    array = np.random.rand(42, 3)

    def test_a(array):
        return tensor_with_grad_a(array, parameter="g")

    def test_b(array):
        return tensor_with_grad_b(array, parameter="g")

    check_finite_differences(test_a, array, parameter="g")
    check_finite_differences(test_b, array, parameter="g")
