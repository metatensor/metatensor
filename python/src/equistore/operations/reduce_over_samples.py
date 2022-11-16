import numpy as np


from equistore import Labels, TensorBlock, TensorMap

from . import _dispatch
from typing import List


def _reduce_over_samples_block(
    block: TensorBlock, sample_names: List[str], reduction: str
) -> TensorBlock:
    """Create a new :py:class:`TensorBlock` summing the ``properties`` among
    the selected ``samples``.
    The output :py:class:`TensorBlocks` have the same components of the input one.
    :param block: -> input block
    :param sample_names: -> names of samples to sum
    """
    for sample in sample_names:
        assert (
            sample in block.samples.names
        ), "the values in sample_names should be contained in block.samples.names"

    if reduction not in [
        "sum",
        "mean",
    ]:
        raise ValueError(
            '_reduce_over_samples_block supports only reduction="sum" or "mean",'
            f'"{reduction}" was passed'
        )

    # get the indices of the selected sample
    sample_selected = [block.samples.names.index(sample) for sample in sample_names]
    # reshaping the samples in a 2D array
    samples = block.samples.view(dtype=np.int32).reshape(block.samples.shape[0], -1)
    s, index = np.unique(samples[:, sample_selected], return_inverse=True, axis=0)
    values_result = _dispatch.zeros_like(
        block.values, shape=(s.shape[0],) + block.values.shape[1:]
    )

    if reduction == "mean":
        bincount = _dispatch.bincount(index)
        _dispatch.index_add(
            values_result,
            block.values[:, ...]
            / bincount[index].reshape(
                (len(index),) + (len(block.values.shape) - 1) * (1,)
            ),
            index,
        )
    elif reduction == "sum":
        _dispatch.index_add(values_result, block.values, index)

    result_block = TensorBlock(
        values=values_result,
        samples=Labels(
            [sample for sample in sample_names],
            np.array([i for i in s], dtype=np.int32),
        ),
        components=block.components,
        properties=block.properties,
    )

    for parameter, gradient in block.gradients():
        # here we need to copy because we want to modify the samples array
        samples = (
            gradient.samples.view(dtype=np.int32)
            .reshape(gradient.samples.shape[0], -1)
            .copy()
        )

        # change the first columns of the samples array with the mapping
        # between samples and gradient.samples
        samples[:, 0] = index[samples[:, 0]]

        s_gradient, index_gradient = np.unique(
            samples[:, :], return_inverse=True, axis=0
        )
        data_result = _dispatch.zeros_like(
            gradient.data, shape=(s_gradient.shape[0],) + gradient.data.shape[1:]
        )

        if reduction == "mean":
            bincount = _dispatch.bincount(index_gradient)
            _dispatch.index_add(
                data_result,
                gradient.data[:, ...]
                / bincount[index_gradient].reshape(
                    (len(index_gradient),) + (len(gradient.data.shape) - 1) * (1,)
                ),
                index_gradient,
            )
        elif reduction == "sum":
            _dispatch.index_add(data_result, gradient.data, index_gradient)

        result_block.add_gradient(
            parameter,
            data_result,
            Labels(
                ["sample"] + [i for i in gradient.samples.names[1:]],
                np.array([i for i in s_gradient], dtype=np.int32),
            ),
            gradient.components,
        )

    return result_block


def sum_over_samples(tensor: TensorMap, sample_names: List[str]) -> TensorMap:
    """Create a new :py:class:`TensorMap` with the same keys as
    as the input `tensor`, and each :py:class:`TensorBlock`
    is obtained summing the corresponding input :py:class:`TensorBlock`
    over the rows with the same sample_names.

    For example if ``sample_name = ["structure"]``, the function sums over all
    the sample with the same structure, and if ``sample_name = ["structure",
    "center"]`` it sums over all the rows with the same values for both
    ``"structure"`` and ``"center"`` samples.

    :param tensor:  input :py:class:`TensorMap`
    :param sample_names: names of samples to sum
    """
    # We check that the names in sample_names are indeed presents in the block sample
    for sample in sample_names:
        if sample not in tensor.sample_names:
            raise ValueError(f"TensorBlocks have not sample={sample}")
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
    as the input `tensor`, and each :py:class:`TensorBlock`
    is obtained averaging the corresponding input :py:class:`TensorBlock`
    over the rows with the same sample_names.

    For example if ``sample_name = ["structure"]``, the function averages over all
    the sample with the same structure, and if ``sample_name = ["structure",
    "center"]`` it averages over all the rows with the same values for both
    ``"structure"`` and ``"center"`` samples.

    :param tensor:  input :py:class:`TensorMap`
    :param sample_names: names of samples to average
    """
    # I check that the names in sample_names are indeed presents in the block sample
    # I check only the first one because all the block have the same sample names
    for sample in sample_names:
        if sample not in tensor[0].samples.names:
            raise ValueError(f"TensorBlocks have not sample={sample}")
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
