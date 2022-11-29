from typing import List

import numpy as np

from equistore import Labels, TensorBlock, TensorMap

from . import _dispatch


def _reduce_over_samples_block(
    block: TensorBlock, sample_names: List[str], reduction: str
) -> TensorBlock:
    """Create a new :py:class:`TensorBlock` summing the ``properties`` among
    the selected ``samples``.
    The output :py:class:`TensorBlocks` have the same components of the input one.
    :param block: -> input block
    :param sample_names: -> names of samples to sum
    """

    block_samples = block.samples
    for sample in sample_names:
        assert sample in block_samples.names

    assert reduction in ["sum", "mean"]

    # get the indices of the selected sample
    sample_selected = [block_samples.names.index(sample) for sample in sample_names]
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

    result_block = TensorBlock(
        values=values_result,
        samples=Labels(
            sample_names,
            new_samples,
        ),
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

        result_block.add_gradient(
            parameter,
            data_result,
            Labels(gradient_samples.names, new_gradient_samples),
            gradient.components,
        )

    return result_block


def sum_over_samples(tensor: TensorMap, sample_names: List[str]) -> TensorMap:
    """Create a new :py:class:`TensorMap` with the same keys as
    as the input ``tensor``, and each :py:class:`TensorBlock` is obtained
    summing the corresponding input :py:class:`TensorBlock` over the rows with
    the same ``sample_names``.

    For example if ``sample_name = ["structure"]``, the function sums over all
    the sample with the same structure, and if ``sample_name = ["structure",
    "center"]`` it sums over all the rows with the same values for both
    ``"structure"`` and ``"center"`` samples.

    :param tensor: input :py:class:`TensorMap`
    :param sample_names: names of samples to sum over
    """
    # We check that the names in sample_names are indeed presents in the block
    for sample in sample_names:
        if sample not in tensor.sample_names:
            raise ValueError(
                f"one of the requested sample name ({sample}) is not part of "
                "this TensorMap"
            )

    blocks = []
    for key, block in tensor:
        blocks.append(
            _reduce_over_samples_block(
                block=block,
                sample_names=sample_names,
                reduction="sum",
            )
        )
    return TensorMap(tensor.keys, blocks)


def mean_over_samples(tensor: TensorMap, sample_names: List[str]) -> TensorMap:
    """Create a new :py:class:`TensorMap` with the same keys as
    as the input ``tensor``, and each :py:class:`TensorBlock` is obtained
    averaging the corresponding input :py:class:`TensorBlock` over the rows with
    the same ``sample_names``.

    For example if ``sample_name = ["structure"]``, the function averages over
    all the sample with the same structure, and if ``sample_name = ["structure",
    "center"]`` it averages over all the rows with the same values for both
    ``"structure"`` and ``"center"`` samples.

    :param tensor: input :py:class:`TensorMap`
    :param sample_names: names of samples to average over
    """
    # We check that the names in sample_names are indeed presents in the block
    for sample in sample_names:
        if sample not in tensor.sample_names:
            raise ValueError(
                f"one of the requested sample name ({sample}) is not part of "
                "this TensorMap"
            )

    blocks = []
    for key, block in tensor:
        blocks.append(
            _reduce_over_samples_block(
                block=block,
                sample_names=sample_names,
                reduction="mean",
            )
        )
    return TensorMap(tensor.keys, blocks)
