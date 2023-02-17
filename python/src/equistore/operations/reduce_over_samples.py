from typing import List, Union

import numpy as np

from equistore import Labels, TensorBlock, TensorMap

from . import _dispatch


def _reduce_over_samples_block(
    block: TensorBlock, remaining_samples: List[str], reduction: str
) -> TensorBlock:
    """
    Create a new :py:class:`TensorBlock` reducing the ``properties`` among the
    selected ``samples``.

    The output :py:class:`TensorBlocks` have the same components of the input
    one. Both "sum" and "mean" reductions can be performed.

    :param block: input block
    :param remaining_samples: names of samples to reduce over
    :param reduction: how to reduce, only available values are "mean" or "sum"
    """

    block_samples = block.samples
    for sample in remaining_samples:
        assert sample in block_samples.names

    assert reduction in ["sum", "mean"]

    # get the indices of the selected sample
    sample_selected = [
        block_samples.names.index(sample) for sample in remaining_samples
    ]
    # reshaping the samples in a 2D array
    samples = block_samples.view(dtype=np.int32).reshape(block_samples.shape[0], -1)
    # get which samples will still be there after reduction
    new_samples, index = np.unique(
        samples[:, sample_selected], return_inverse=True, axis=0
    )

    block_values = block.values
    other_shape = block_values.shape[1:]
    values_result = _dispatch.zeros_like(
        block_values, shape=(new_samples.shape[0],) + other_shape
    )

    _dispatch.index_add(
        values_result,
        block_values,
        index,
    )

    if reduction == "mean":
        bincount = _dispatch.bincount(index)
        values_result = values_result / bincount.reshape(
            (-1,) + (1,) * len(other_shape)
        )

    # check if the reduce operation reduce all the samples
    if len(remaining_samples) == 0:
        samples_label = Labels.single()
    else:
        samples_label = Labels(
            remaining_samples,
            new_samples,
        )

    result_block = TensorBlock(
        values=values_result,
        samples=samples_label,
        components=block.components,
        properties=block.properties,
    )

    for parameter, gradient in block.gradients():
        gradient_samples = gradient.samples
        # here we need to copy because we want to modify the samples array
        samples = (
            gradient_samples.view(dtype=np.int32)
            .reshape(gradient_samples.shape[0], -1)
            .copy()
        )

        # change the first columns of the samples array with the mapping
        # between samples and gradient.samples
        samples[:, 0] = index[samples[:, 0]]

        new_gradient_samples, index_gradient = np.unique(
            samples[:, :], return_inverse=True, axis=0
        )

        gradient_data = gradient.data
        other_shape = gradient_data.shape[1:]
        data_result = _dispatch.zeros_like(
            gradient_data,
            shape=(new_gradient_samples.shape[0],) + other_shape,
        )
        _dispatch.index_add(data_result, gradient_data, index_gradient)

        if reduction == "mean":
            bincount = _dispatch.bincount(index_gradient)
            data_result = data_result / bincount.reshape(
                (-1,) + (1,) * len(other_shape)
            )

        # no check for the len of the gradient sample is needed becouse there always
        # will be at least one sample in the gradient

        result_block.add_gradient(
            parameter,
            data_result,
            Labels(gradient_samples.names, new_gradient_samples),
            gradient.components,
        )

    return result_block


def _reduce_over_samples(
    tensor: TensorMap, samples_names: Union[List[str], str], reduction: str
) -> TensorMap:
    """
    Create a new :py:class:`TensorMap` with the same keys as as the input
    ``tensor``, and each :py:class:`TensorBlock` is obtained summing the
    corresponding input :py:class:`TensorBlock` over the ``samples_names``
    indices.

    Both "sum" and "mean" reductions can be performed.

    :param tensor: input :py:class:`TensorMap`
    :param samples_names: names of samples to reduce over
    :param reduction: how to reduce, only available values are "mean" or "sum"
    """
    if isinstance(samples_names, str):
        samples_names = [samples_names]

    for sample in samples_names:
        if sample not in tensor.sample_names:
            raise ValueError(
                f"one of the requested sample name ({sample}) is not part of "
                "this TensorMap"
            )

    remaining_samples = [
        s_name for s_name in tensor.sample_names if s_name not in samples_names
    ]

    blocks = []
    for _, block in tensor:
        blocks.append(
            _reduce_over_samples_block(
                block=block,
                remaining_samples=remaining_samples,
                reduction=reduction,
            )
        )
    return TensorMap(tensor.keys, blocks)


def sum_over_samples(
    tensor: TensorMap, samples_names: Union[List[str], str]
) -> TensorMap:
    """
    Create a new :py:class:`TensorMap` with the same keys as as the input
    ``tensor``, and each :py:class:`TensorBlock` is obtained summing the
    corresponding input :py:class:`TensorBlock` over the ``samples_names``
    indices.

    ``samples_names`` indicates over which variables in the samples the sum is
    performed. It accept either a single string or a list of the string with the
    sample names corresponding to the directions along which the sum is
    performed. A single string is equivalent to a list with a single element:
    ``samples_names = "center"`` is the same as ``samples_names = ["center"]``.

    Here is an example using this function

    >>> block = TensorBlock(
    ...     values=np.array([
    ...         [1, 2, 4],
    ...         [3, 5, 6],
    ...         [7, 8, 9],
    ...         [10, 11, 12],
    ...     ]),
    ...     samples=Labels(
    ...         ["structure", "center"],
    ...         np.array([
    ...             [0, 0],
    ...             [0, 1],
    ...             [1, 0],
    ...             [1, 1],
    ...         ]),
    ...     ),
    ...     components=[],
    ...     properties=Labels(
    ...         ["properties"], np.array([[0], [1], [2]])
    ...     ),
    ... )
    >>> keys = Labels(names=["key"], values=np.array([[0]]))
    ...
    >>> tensor = TensorMap(keys, [block])
    ...
    >>> tensor_sum = sum_over_samples(tensor, samples_names="center")
    ...
    >>> # only 'structure' is left as a sample
    >>> print(tensor_sum.block(0))
    TensorBlock
        samples (2): ['structure']
        components (): []
        properties (3): ['properties']
        gradients: no
    >>> print(tensor_sum.block(0).samples)
    [(0,) (1,)]
    >>> print(tensor_sum.block(0).values)
    [[ 4  7 10]
     [17 19 21]]


    :param tensor: input :py:class:`TensorMap`
    :param samples_names: names of samples to sum over
    """

    return _reduce_over_samples(
        tensor=tensor, samples_names=samples_names, reduction="sum"
    )


def mean_over_samples(tensor: TensorMap, samples_names: List[str]) -> TensorMap:
    """Create a new :py:class:`TensorMap` with the same keys as
    as the input ``tensor``, and each :py:class:`TensorBlock` is obtained
    averaging the corresponding input :py:class:`TensorBlock` over the ``samples_names``
    indices.

    ``samples_names`` indicates over which variables in the samples the mean is
    performed. It accept either a single string or a list of the string with the
    sample names corresponding to the directions along which the mean is performed.
    A single string is equivalent to a list with a single element:
    ``samples_names = "center"`` is the same as ``samples_names = ["center"]``.

    For an usage example see the doc for ``sum_over_samples``.

    :param tensor: input :py:class:`TensorMap`
    :param samples_names: names of samples to average over
    """
    return _reduce_over_samples(
        tensor=tensor, samples_names=samples_names, reduction="mean"
    )
