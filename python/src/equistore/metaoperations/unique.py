"""
Module for finding unique metadata indices for TensorMaps and TensorBlocks
"""
from typing import List, Tuple, Union

import numpy as np

from ..block import TensorBlock
from ..labels import Labels
from ..tensor import TensorMap


def unique(
    tensor: TensorMap,
    axis: str,
    names: Union[List[str], Tuple[str], str],
) -> Labels:
    """
    For a given ``axis`` (either "samples" or "properties"), and for the given
    samples/proeprties ``names``, returns a :py:class:`Labels` object of the
    unique indices in the input :py:class:`TensorMap` ``tensor``.

    If there are no indices in ``tensor`` corresponding to the specified
    ``axis`` and ``names``, an empty Labels object with the correct names as in
    the passed ``names`` is returned.

    :param block: the :py:class:`TensorMap` to find unique indices for.
    :param axis: a str, either "samples" or "properties", corresponding to the
        axis along which the named unique indices should be found.
    :param names: a str, list of str, or tuple of str corresponding to the
        name(s) of the indices along the specified ``axis`` for which the unique
        values should be found.

    :return: a sorted :py:class:`Labels` object containing the unique indices
        for the input ``tensor``. Each element in the returned
        :py:class:`Labels` object has len(names) entries.
    """
    # Parse input args
    if not isinstance(tensor, TensorMap):
        raise TypeError("``tensor`` must be an equistore TensorMap")
    names = (
        [names]
        if isinstance(names, str)
        else (list(names) if isinstance(names, tuple) else names)
    )
    _check_args(tensor, axis, names)

    return _unique_from_blocks(tensor.blocks(), axis, names)


def unique_block(
    block: TensorBlock,
    axis: str,
    names: Union[List[str], Tuple[str], str],
) -> Labels:
    """
    For a given ``axis`` (either "samples" or "properties"), and for the given
    samples/proeprties ``names``, returns a :py:class:`Labels` object of the
    unique indices in the input :py:class:`TensorBlock` ``block``.

    If there are no indices in ``block`` corresponding to the specified ``axis``
    and ``names``, an empty Labels object with the correct names as in the
    passed ``names`` is returned.

    :param block: the :py:class:`TensorBlock` to find unique indices for.
    :param axis: a str, either "samples" or "properties", corresponding to the
        axis along which the named unique indices should be found.
    :param names: a str, list of str, or tuple of str corresponding to the
        name(s) of the indices along the specified ``axis`` for which the unique
        values should be found.

    :return: a sorted :py:class:`Labels` object containing the unique indices
        for the input ``block``. Each element in the returned :py:class:`Labels`
        object has len(names) entries.
    """
    # Parse input args
    if not isinstance(block, TensorBlock):
        raise TypeError("``block`` must be an equistore TensorBlock")
    names = (
        [names]
        if isinstance(names, str)
        else (list(names) if isinstance(names, tuple) else names)
    )
    _check_args(block, axis, names)

    return _unique_from_blocks([block], axis, names)


def _unique_from_blocks(
    blocks: List[TensorBlock], axis: str, names: List[str]
) -> Labels:
    """
    Finds the unique indices of a list of blocks along the given ``axis`` and
    for the specified ``names``
    """
    # Extract indices from each block
    all_idxs = []
    for block in blocks:
        idxs = block.samples[names] if axis == "samples" else block.properties[names]
        for idx in idxs:
            all_idxs.append(idx)

    # If no matching indices across all blocks return a empty Labels w/ the
    # correct names
    if len(all_idxs) == 0:
        # Create Labels with single entry
        labels = Labels(names=names, values=np.array([[i for i in range(len(names))]]))
        # rslice to zero length
        return labels[:0]

    # Define the unique and sorted indices
    unique_idxs = np.unique(all_idxs, axis=0)

    # Return as Labels
    return Labels(names=names, values=np.array([[j for j in i] for i in unique_idxs]))


def _check_args(
    tensor: Union[TensorMap, TensorBlock],
    axis: str,
    names: List[str],
):
    """Checks input args for :py:func:`unique_indices`."""
    # Check tensors
    if isinstance(tensor, TensorMap):
        blocks = tensor.blocks()
    elif isinstance(tensor, TensorBlock):
        blocks = [tensor]
    # Check axis
    if not isinstance(axis, str):
        raise TypeError("``axis`` must be a str, either 'samples' or 'properties'")
    if axis not in ["samples", "properties"]:
        raise ValueError("``axis`` must be passsed as either 'samples' or 'properties'")
    # Check names
    if not isinstance(names, list):
        raise TypeError("``names`` must be a list of str")
    for block in blocks:
        tmp_names = block.samples.names if axis == "samples" else block.properties.names
        for name in names:
            if name not in tmp_names:
                raise ValueError(
                    "the block(s) passed must have samples/properties"
                    + " names that matches the one passed in ``names``"
                )
