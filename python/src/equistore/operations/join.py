from typing import List
import functools
import operator

import numpy as np

from ..labels import Labels
from ..tensor import TensorMap, TensorBlock
from .equal_metadata import _check_maps


def join(tensor_maps: List[TensorMap], axis: str):
    """Join a sequence of :py:class:`TensorMap` along an axis.

    The ``axis`` parameter specifies the type join. For example, if
    ``axis='properties'`` it will be the `tensor_maps` will be joined along the
    `properties` dimension and for ``axis='samples'`` they will be the along the
    samples dimension.

    `join` will create an additional label `tensor` specifiying the original
    index in the list of `tensor_maps`.

    :param tensormaps:
        sequence of :py:class:`TensorMap` for join
    :param axis:
        A string indicating how the tensormaps are stacked. Allowed
        values are ``'properties'`` or ``'samples'``.

    :return tensor_joined:
        The stacked :py:class:`TensorMap` with more properties or samples
        than the input TensorMap.
    """

    if not isinstance(tensor_maps, (list, tuple)):
        raise TypeError(
            "the `TensorMap`s to join must be provided as a list or a tuple"
        )

    if len(tensor_maps) < 1:
        raise ValueError("provide at least one `TensorMap` for joining")

    if axis == "samples":
        common_labels_names = _get_common_labels_names()
    elif axis == "properties":
        pass
    if axis not in ("samples", "properties"):
        raise ValueError(
            "Only `'properties'` or `'samples'` are "
            "valid values for the `axis` parameter."
        )

    if len(tensor_maps) == 1:
        return tensor_maps[0]

    for ts_to_join in tensor_maps[1:]:
        _check_maps(tensor_maps[0], ts_to_join, "join")

    keys_names = tensor_maps[0].keys.names + ("tensor",)
    keys_values = []
    blocks = []

    for i, tensor in enumerate(tensor_maps):
        keys_values += [value + (i,) for value in tensor.keys.tolist()]

        for _, block in tensor:
            if axis == "samples":
                block = TensorBlock(
                    block.values,
                    block.samples,
                    block.components,
                    block.properties,
                )
            else:
                block = TensorBlock(
                    block.values,
                    block.samples,
                    block.components,
                    block.properties,
                )

            blocks.append(block)

    keys = Labels(names=keys_names, values=np.array(keys_values))
    tensor = TensorMap(keys=keys, blocks=blocks)

    if axis == "samples":
        tensor_joined = tensor.keys_to_samples("tensor")
    else:
        tensor_joined = tensor.keys_to_properties("tensor")

    # TODO: once we have functions to manipulate meta data we can try to
    # remove the `tensor` label entry after joining.

    return tensor_joined

def _get_common_str(labels: List[List[str]) -> List[str]:
    names_list = [label.names for label in labels]

    # We use functools to flatten a list of sublists::
    #
    #   [('a', 'b', 'c'), ('a', 'b')] -> ['a', 'b', 'c', 'a', 'b']
    #
    # A nested list with sublist of different shapes can not be handled by np.unique.
    unique_names = np.unique(functools.reduce(operator.concat, names_list))

    # Label names are unique: We can do an equal check only checking the lengths.
    names_are_same = np.all(
        len(unique_names) == np.array([len(names) for names in names_list])
    )

    if names_are_same:
        return names_list[0]
    else:
        return ["property"]


def _join_labels(labels: List[Labels]) -> Labels:
    """Join a sequence of :py:class:`Labels`"""
    names_list = [label.names for label in labels]
    values_list = [label.tolist() for label in labels]
    tensor_values = np.repeat(
        a=np.arange(len(values_list)), repeats=[len(value) for value in values_list]
    )

    # We use functools to flatten a list of sublists::
    #
    #   [('a', 'b', 'c'), ('a', 'b')] -> ['a', 'b', 'c', 'a', 'b']
    #
    # A nested list with sublist of different shapes can not be handled by np.unique.
    unique_names = np.unique(functools.reduce(operator.concat, names_list))

    # Label names are unique: We can do an equal check only checking the lengths.
    names_are_same = np.all(
        len(unique_names) == np.array([len(names) for names in names_list])
    )

    if names_are_same:
        unique_values = np.unique(np.vstack(values_list), axis=0)
        values_are_unique = len(unique_values) == len(np.vstack(values_list))
        if values_are_unique:
            new_names = names_list[0]
            new_values = np.vstack(values_list)
        else:
            new_names = list(names_list[0])
            new_values = np.vstack(values_list)
    else:
        new_names = ["property"]
        new_values = np.hstack([np.arange(len(values)) for values in values_list])

    return Labels(names=new_names, values=new_values)
