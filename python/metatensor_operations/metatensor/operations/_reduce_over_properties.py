from typing import List, Optional, Union

import numpy as np

from . import _dispatch
from ._backend import (
    Labels,
    TensorBlock,
    TensorMap,
    torch_jit_is_scripting,
    torch_jit_script,
)


def _reduce_over_properties_block(
    block: TensorBlock,
    property_names: Union[List[str], str],
    reduction: str,
    remaining_properties: Optional[List[str]] = None,
) -> TensorBlock:
    """
    Create a new :py:class:`TensorBlock` reducing the ``properties`` among the
    selected ``properties``.

    The output :py:class:`TensorBlocks` have the same components as the input.
    "sum", "mean", "std" or "var" reductions can be performed.

    :param block:
        input block
    :param property_names:
        names of properties to reduce over. it is ignored if remaining_properties is
        given
    :param remaining_properties:
        names of properties that should remain after reducing reduce over.
        it is computed automatically from property_names if missing or set to None
    :param reduction:
        how to reduce, only available values are "mean", "sum", "std" or "var"
    """
    if isinstance(property_names, str):
        property_names_list = [property_names]
    else:
        property_names_list = property_names

    block_properties = block.properties

    remaining_properties = None
    if remaining_properties is None:
        remaining_property_names: List[str] = []
        for p_name in block_properties.names:
            if p_name in property_names_list:
                continue
            remaining_property_names.append(p_name)
    else:
        remaining_property_names = remaining_properties

    for sample in remaining_property_names:
        assert sample in block_properties.names

    assert reduction in ["sum", "mean", "var", "std"]
    # get the indices of the selected property
    remaining_property_columns = [
        block_properties.names.index(prop) for prop in remaining_property_names
    ]

    if len(block.properties) == 0:
        properties = Labels(
            remaining_property_names,
            _dispatch.zeros_like(block.values, [0, len(remaining_property_names)]),
        )

        result_block = TensorBlock(
            values=block.values,
            samples=block.samples,
            components=block.components,
            properties=properties,
        )

        # The gradient does not change because the only thing that matters for
        # the gradients are the samples to which they are connected, but in this
        # case there are no samples in the TensorBlock
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

    # get which properties will still be there after reduction
    if len(remaining_property_names) == 0:
        new_properties = _dispatch.zeros_like(block_properties.values, shape=(1, 0))
        index = _dispatch.zeros_like(
            block_properties.values, shape=(block_properties.values.shape[0],)
        )
    else:
        new_properties, index = _dispatch.unique_with_inverse(
            block_properties.values[:, remaining_property_columns], axis=0
        )
        index = index.reshape(-1)

    block_values = block.values
    shape = (block_values.shape)[:-1] + (new_properties.shape[0],)
    values_sum = _dispatch.zeros_like(block_values, shape=shape)

    _dispatch.columns_add(values_sum, block_values, index)

    values_mean = _dispatch.empty_like(values_sum, [0])
    values_result = _dispatch.empty_like(values_sum, [0])

    if reduction == "mean" or reduction == "std" or reduction == "var":
        bincount = _dispatch.make_like(_dispatch.bincount(index), values_sum)
        bin_shape = list(block_values.shape)
        bin_shape = [(-1 if i == 1 else 1) for i in range(len(shape))]
        bincount = bincount.reshape(bin_shape)
        values_mean = values_sum / bincount

        if reduction == "std" or reduction == "var":
            values_var = _dispatch.zeros_like(block_values, shape=shape)
            _dispatch.columns_add(
                values_var,
                (block_values - _dispatch.slice_last_dim(values_mean, index)) ** 2,
                index,
            )
            values_var = values_var / bincount

            if reduction == "std":
                values_result = _dispatch.sqrt(values_var)
            elif reduction == "var":
                values_result = values_var

        elif reduction == "mean":
            values_result = values_mean
    elif reduction == "sum":
        values_result = values_sum

    if len(remaining_property_names) == 0:
        properties_label = Labels(
            names="_",
            values=_dispatch.zeros_like(block_properties.values, shape=(1, 1)),
        )
    else:
        properties_label = Labels(
            remaining_property_names,
            new_properties,
        )

    result_block = TensorBlock(
        values=values_result,
        samples=block.samples,
        components=block.components,
        properties=properties_label,
    )

    for parameter, gradient in block.gradients():
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        # check if all gradients are zeros
        if len(gradient.properties) == 0:
            # The gradients does not change because, if they are all zeros, the
            # gradients after reducing operation is still zero.

            # For any function of the TensorBlock values x(t):
            # f(x(t))-> df(x(t))/dx * dx/dt
            # and dx/dt == 0.
            result_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=gradient.properties,
                ),
            )
            continue

        new_gradient_properties, index_gradient = result_block.properties.values, index
        index_gradient = index_gradient.reshape(-1)

        gradient_values = gradient.values
        other_shape = gradient_values.shape[:-1]
        gradient_values_result = _dispatch.zeros_like(
            gradient_values,
            shape=other_shape + (new_gradient_properties.shape[0],),
        )
        _dispatch.columns_add(gradient_values_result, gradient_values, index_gradient)

        if reduction == "mean" or reduction == "var" or reduction == "std":
            bincount = _dispatch.bincount(index_gradient)
            bincount = bincount.reshape((1,) * len(other_shape) + (-1,))
            gradient_values_result = gradient_values_result / bincount
            if reduction == "std" or reduction == "var":
                values_times_gradient_values = _dispatch.zeros_like(gradient_values)

                for i in range(gradient.samples.values.shape[0]):
                    s = gradient.samples.entry(i)
                    values_times_gradient_values[i] = (
                        gradient_values[i] * block_values[int(s[0])]
                    )

                values_grad_result = _dispatch.zeros_like(
                    gradient_values,
                    shape=other_shape + (new_gradient_properties.shape[0],),
                )
                _dispatch.columns_add(
                    values_grad_result,
                    values_times_gradient_values,
                    index_gradient,
                )

                values_grad_result = values_grad_result / bincount
                sample_indices = gradient.samples.values[:, 0]
                if reduction == "var":
                    for i, _ in enumerate(new_gradient_properties):
                        shape = (-1,) + (1,) * (gradient_values_result.ndim - 2)
                        gradient_values_result = _dispatch.scatter_last_dim(
                            gradient_values_result,
                            i,
                            _dispatch.slice_last_dim(gradient_values_result, i)
                            * _dispatch.slice_last_dim(values_mean, i)[
                                sample_indices
                            ].reshape(shape),
                        )
                    gradient_values_result = 2 * (
                        values_grad_result - gradient_values_result
                    )
                else:  # std
                    for i, _ in enumerate(new_gradient_properties):
                        shape = (-1,) + (1,) * (gradient_values_result.ndim - 2)
                        if torch_jit_is_scripting():
                            gradient_values_result = _dispatch.scatter_last_dim(
                                gradient_values_result,
                                i,
                                _dispatch.slice_last_dim(gradient_values_result, i)
                                * _dispatch.slice_last_dim(values_mean, i)[
                                    sample_indices
                                ].reshape(shape)
                                / _dispatch.slice_last_dim(values_result, i)[
                                    sample_indices
                                ].reshape(shape),
                            )
                        else:
                            # no need to be torchscript compatible, let's keep it simple
                            # only numpy raise a warning for division by zero
                            with np.errstate(divide="ignore", invalid="ignore"):
                                gradient_values_result[..., i] = (
                                    values_grad_result[..., i]
                                    - (
                                        gradient_values_result[..., i]
                                        * values_mean[..., i][sample_indices].reshape(
                                            shape
                                        )
                                    )
                                ) / values_result[..., i][sample_indices].reshape(shape)
                        gradient_values_result = _dispatch.scatter_last_dim(
                            gradient_values_result,
                            i,
                            _dispatch.nan_to_num(
                                _dispatch.slice_last_dim(gradient_values_result, i),
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            ),
                        )
                        gradient_values_result[..., i] = _dispatch.nan_to_num(
                            gradient_values_result[..., i],
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )

        # no check for the len of the gradient sample is needed because there
        # always will be at least one sample in the gradient
        result_block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=gradient_values_result,
                samples=gradient.samples,
                components=gradient.components,
                properties=result_block.properties,
            ),
        )

    return result_block


def _reduce_over_properties(
    tensor: TensorMap, property_names: Union[List[str], str], reduction: str
) -> TensorMap:
    """
    Create a new :py:class:`TensorMap` with the same keys as as the input
    ``tensor``, and each :py:class:`TensorBlock` is obtained summing the
    corresponding input :py:class:`TensorBlock` over the ``property_names``
    indices.

    "sum", "mean", "std" or "var" reductions can be performed.

    :param tensor: input :py:class:`TensorMap`
    :param property_names: names of properties to reduce over
    :param reduction: how to reduce, only available values are "mean", "sum",
    "std" or "var"
    """
    if isinstance(property_names, str):
        property_names_list = [property_names]
    else:
        property_names_list = property_names

    for property in property_names_list:
        if property not in tensor.property_names:
            raise ValueError(
                f"one of the requested property name ({property}) is not part of "
                "this TensorMap"
            )

    remaining_properties: List[str] = []
    for p_name in tensor.property_names:
        if p_name in property_names_list:
            continue
        remaining_properties.append(p_name)

    blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        blocks.append(
            _reduce_over_properties_block(
                block=block,
                property_names=property_names_list,
                reduction=reduction,
                remaining_properties=remaining_properties,
            )
        )
    return TensorMap(tensor.keys, blocks)


@torch_jit_script
def sum_over_properties_block(
    block: TensorBlock, property_names: Union[List[str], str]
) -> TensorBlock:
    """
    Sum a :py:class:`TensorBlock`, combining the properties according to
    ``property_names``.

    This function creates a new :py:class:`TensorBlock` in which each property is
    obtained summing over the ``property_names`` indices, so that the resulting
    :py:class:`TensorBlock` does not have those indices.

    ``property_names`` indicates over which dimensions in the properties the sum is
    performed. It accept either a single string or a list of the string with the
    property names corresponding to the directions along which the sum is performed. A
    single string is equivalent to a list with a single element: ``property_names =
    "i"`` is the same as ``property_names = ["i"]``.

    :param block:
        input :py:class:`TensorBlock`
    :param property_names:
        names of properties to sum over

    :returns:
        a :py:class:`TensorBlock` containing the reduced values and property labels

    >>> from metatensor import Labels, TensorBlock, TensorMap
    >>> block = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1, 3, 7, 10],
    ...             [2, 5, 8, 11],
    ...             [4, 6, 9, 12],
    ...         ]
    ...     ),
    ...     samples=Labels(
    ...         ["system", "atom"],
    ...         np.array(
    ...             [
    ...                 [0, 0],
    ...                 [0, 1],
    ...                 [1, 0],
    ...             ]
    ...         ),
    ...     ),
    ...     components=[],
    ...     properties=Labels(
    ...         ["i", "j"],
    ...         np.array(
    ...             [
    ...                 [0, 0],
    ...                 [0, 1],
    ...                 [1, 0],
    ...                 [1, 1],
    ...             ]
    ...         ),
    ...     ),
    ... )
    >>> block_sum = sum_over_properties_block(block, property_names="j")
    >>> print(block_sum.properties)
    Labels(
        i
        0
        1
    )
    >>> print(block_sum.values)
    [[ 4 17]
     [ 7 19]
     [10 21]]
    """

    return _reduce_over_properties_block(
        block=block, property_names=property_names, reduction="sum"
    )


@torch_jit_script
def sum_over_properties(
    tensor: TensorMap, property_names: Union[List[str], str]
) -> TensorMap:
    """
    Sum a :py:class:`TensorMap`, combining the properties according to
    ``property_names``.

    This function creates a new :py:class:`TensorMap` with the same keys
    as as the input ``tensor``. Each :py:class:`TensorBlock` is obtained summing the
    corresponding input :py:class:`TensorBlock` over the ``property_names``
    indices, essentially calling :py:func:`sum_over_properties_block` over each
    block in ``tensor``.

    ``property_names`` indicates over which dimensions in the properties the sum is
    performed. It accept either a single string or a list of the string with the
    property names corresponding to the directions along which the sum is
    performed. A single string is equivalent to a list with a single element:
    ``property_names = "atom"`` is the same as ``property_names = ["atom"]``.

    :param tensor:
        input :py:class:`TensorMap`
    :param property_names:
        names of properties to sum over

    :returns:
        a :py:class:`TensorMap` containing the reduced values and property labels

    >>> from metatensor import Labels, TensorBlock, TensorMap
    >>> block = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1, 3, 7, 10],
    ...             [2, 5, 8, 11],
    ...             [4, 6, 9, 12],
    ...         ]
    ...     ),
    ...     samples=Labels(
    ...         ["system", "atom"],
    ...         np.array(
    ...             [
    ...                 [0, 0],
    ...                 [0, 1],
    ...                 [1, 0],
    ...             ]
    ...         ),
    ...     ),
    ...     components=[],
    ...     properties=Labels(
    ...         ["i", "j"],
    ...         np.array(
    ...             [
    ...                 [0, 0],
    ...                 [0, 1],
    ...                 [1, 0],
    ...                 [1, 1],
    ...             ]
    ...         ),
    ...     ),
    ... )
    >>> keys = Labels(names=["key"], values=np.array([[0]]))
    >>> tensor = TensorMap(keys, [block])
    >>> tensor_sum = sum_over_properties(tensor, property_names="j")
    >>> # only 'i' is left as a property
    >>> print(tensor_sum.block(0))
    TensorBlock
        samples (3): ['system', 'atom']
        components (): []
        properties (2): ['i']
        gradients: None
    >>> print(tensor_sum.block(0).properties)
    Labels(
        i
        0
        1
    )
    >>> print(tensor_sum.block(0).values)
    [[ 4 17]
     [ 7 19]
     [10 21]]
    """

    return _reduce_over_properties(
        tensor=tensor, property_names=property_names, reduction="sum"
    )


@torch_jit_script
def mean_over_properties_block(
    block: TensorBlock, property_names: Union[List[str], str]
) -> TensorBlock:
    """Averages a :py:class:`TensorBlock`, combining the properties according
    to ``property_names``.

    See also :py:func:`sum_over_properties_block` and :py:func:`mean_over_properties`

    :param block:
        input :py:class:`TensorBlock`
    :param property_names:
        names of properties to average over

    :returns:
        a :py:class:`TensorBlock` containing the reduced values and property labels
    """

    return _reduce_over_properties_block(
        block=block, property_names=property_names, reduction="mean"
    )


@torch_jit_script
def mean_over_properties(
    tensor: TensorMap, property_names: Union[str, List[str]]
) -> TensorMap:
    """Compute the mean of a :py:class:`TensorMap`, combining the properties according
    to ``property_names``.

    This function creates a new :py:class:`TensorMap` with the same keys as
    the input ``tensor``, and each :py:class:`TensorBlock` is obtained
    averaging the corresponding input :py:class:`TensorBlock` over the
    ``property_names`` indices.

    ``property_names`` indicates over which dimensions in the properties the mean is
    performed. It accept either a single string or a list of the string with the
    property names corresponding to the directions along which the mean is performed.
    A single string is equivalent to a list with a single element:
    ``property_names = "atom"`` is the same as ``property_names = ["atom"]``.

    For a general discussion of reduction operations and a usage example see the
    doc for :py:func:`sum_over_properties`.

    :param tensor: input :py:class:`TensorMap`
    :param property_names: names of properties to average over
    """

    return _reduce_over_properties(
        tensor=tensor, property_names=property_names, reduction="mean"
    )


@torch_jit_script
def std_over_properties_block(
    block: TensorBlock, property_names: Union[List[str], str]
) -> TensorBlock:
    """Computes the standard deviation for a :py:class:`TensorBlock`,
    combining the properties according to ``property_names``.

    See also :py:func:`sum_over_properties_block` and :py:func:`std_over_properties`

    :param block:
        input :py:class:`TensorBlock`
    :param property_names:
        names of properties to compute the standard deviation for

    :returns:
        a :py:class:`TensorBlock` containing the reduced values and property labels
    """

    return _reduce_over_properties_block(
        block=block, property_names=property_names, reduction="std"
    )


@torch_jit_script
def std_over_properties(
    tensor: TensorMap, property_names: Union[str, List[str]]
) -> TensorMap:
    r"""Compute the standard deviation of a :py:class:`TensorMap`, combining the
    properties according to ``property_names``.

    This function creates a new :py:class:`TensorMap` with the same keys as
    as the input ``tensor``, and each :py:class:`TensorBlock` is obtained
    performing the std deviation of the corresponding input :py:class:`TensorBlock`
    over the ``property_names`` indices.

    ``property_names`` indicates over which dimensions in the properties the mean is
    performed. It accept either a single string or a list of the string with the
    property names corresponding to the directions along which the mean is performed.
    A single string is equivalent to a list with a single element:
    ``property_names = "i"`` is the same as ``property_names = ["i"]``.

    For a general discussion of reduction operations and a usage example see the
    doc for :py:func:`sum_over_properties()`.

    The gradient is implemented as follows:

    .. math::

        \nabla[Std(X)] = 0.5(\nabla[Var(X)])/Std(X)
        = (E[X \nabla X] - E[X]E[\nabla X])/Std(X)

    :param tensor: input :py:class:`TensorMap`
    :param property_names: names of properties to perform the standart deviation over
    """

    return _reduce_over_properties(
        tensor=tensor, property_names=property_names, reduction="std"
    )


@torch_jit_script
def var_over_properties_block(
    block: TensorBlock, property_names: Union[List[str], str]
) -> TensorBlock:
    """Computes the variance for a :py:class:`TensorBlock`,
    combining the properties according to ``property_names``.

    See also :py:func:`sum_over_properties_block` and :py:func:`std_over_properties`

    :param block:
        input :py:class:`TensorBlock`
    :param property_names:
        names of properties to compute the variance for

    :returns:
        a :py:class:`TensorBlock` containing the reduced values and property labels
    """

    return _reduce_over_properties_block(
        block=block, property_names=property_names, reduction="var"
    )


@torch_jit_script
def var_over_properties(
    tensor: TensorMap, property_names: Union[str, List[str]]
) -> TensorMap:
    r"""Compute the variance of a :py:class:`TensorMap`, combining the
    properties according to ``property_names``.

    This function creates a new :py:class:`TensorMap` with the same keys as
    as the input ``tensor``, and each :py:class:`TensorBlock` is obtained
    performing the variance of the corresponding input :py:class:`TensorBlock`
    over the ``property_names`` indices.

    ``property_names`` indicates over which dimensions in the properties the mean is
    performed. It accept either a single string or a list of the string with the
    property names corresponding to the directions along which the mean is performed.
    A single string is equivalent to a list with a single element:
    ``property_names = "i"`` is the same as ``property_names = ["i"]``.

    For a general discussion of reduction operations and a usage example see the
    doc for :py:func:`sum_over_properties`.

    The gradient is implemented as follow:

    .. math::

        \nabla[Var(X)] = 2(E[X \nabla X] - E[X]E[\nabla X])

    :param tensor: input :py:class:`TensorMap`
    :param property_names: names of properties to perform the variance over
    """

    return _reduce_over_properties(
        tensor=tensor, property_names=property_names, reduction="var"
    )
