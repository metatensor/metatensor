"""
Module containing a collate function for use in a Dataloader.
"""

from collections import namedtuple
from typing import List, NamedTuple, Optional, Union

import torch

import metatensor
from metatensor import TensorMap


def group(batch: List[NamedTuple]) -> NamedTuple:
    """
    Collates a minibatch by grouping the data for each data field and returning
    a named tuple.

    `batch` is a list of named tuples. Each is composed of a number of named
    data fields, which are arbitrary objects, such as :py:class:`torch.Tensor`,
    :py:class:`atomistic.Systems`, or :py:class:`TensorMap`.

    Returned is a new named tuple with the same named data fields as the each
    sample in the `batch`, but with the sample data collated for each respective
    field. The indices of the samples in the minibatch belong to the first field
    unpacked from the named tuple, i.e. "sample_indices".

    :param batch: list of named tuples for each sample in the minibatch.

    :return: a named tuple, with the named fields the same as in the original
        samples in the batch, but with the samples grouped by data field.
    """
    return namedtuple("Batch", batch[0]._fields)(*list(zip(*batch)))


def group_and_join(
    batch: List[NamedTuple],
    fields_to_join: Optional[List[str]] = None,
    join_kwargs: Optional[dict] = None,
) -> NamedTuple:
    """
    Collates a minibatch by grouping the data for each fields, joining tensors
    along the samples axis, and returning a named tuple.

    Similar in functionality to the generic :py:func:`group`, but instead data
    fields that are :py:class:`torch.Tensor` objects are vertically stacked, and
    :py:class:`TensorMap` objects are joined along the samples axis.

    `batch` is a list of named tuples. Each has a number of fields that
    correspond to different named data fields. These data fields can be
    arbitrary objects, such as :py:class:`torch.Tensor`,
    :py:class:`atomistic.Systems`, or :py:class:`TensorMap`.

    For each data field, the data object from each sample in the batch is
    collated into a list, except where the data field is a list of
    :py:class:`torch.Tensor` or :py:class:`TensorMap` objects. In this case, the
    tensors are joined along the samples axis. If torch tensors, all must be of
    the same size. In the case of TensorMaps, the union of sparse keys are
    taken.

    Returned is a new named tuple with the same fields as the each sample in the
    `batch`, but with the sample data collated for each respective field. The
    sample indices in the minibatch are in the first field of the named tuple
    under "sample_indices".

    :param batch: list of named tuples for each sample in the batch.
    :param fields_to_join: list of data field names to join. If None, all fields
        that can be joined are joined, i.e. those comprised of
        :py:class:`torch.Tensor` or :py:class:`TensorMap` objects. Any names
        passed that are either invalid or are names of fields that aren't these
        types will be silently ignored.
    :param join_kwargs: keyword arguments passed to the
        :py:func:`metatensor.join` function, to be used when joining data fields
        comprised of :py:class:`TensorMap` objects. If none, the defaults are
        used - see the function documentation for details. The `axis="samples"`
        arg is set by default.

    :return: a named tuple, with the named fields the same as in the original
        samples in the batch, but with the samples collated for each respective
        field. If the data fields are :py:class:`torch.Tensor` or
        :py:class:`TensorMap` objects, they are joined along the samples axis.
    """
    data: List[Union[TensorMap, torch.Tensor]] = []
    names = batch[0]._fields
    if fields_to_join is None:
        fields_to_join = names
    if join_kwargs is None:
        join_kwargs = {}
    for name, field in zip(names, list(zip(*batch))):
        if name == "sample_id":  # special case, keep as is
            data.append(field)
            continue

        if name in fields_to_join:  # Join tensors if requested
            if isinstance(field[0], TensorMap):
                # metatensor.TensorMap type
                data.append(metatensor.join(field, axis="samples", **join_kwargs))
            elif isinstance(field[0], torch.ScriptObject) and field[0]._has_method(
                "keys_to_properties"
            ):  # inferred metatensor.torch.TensorMap type
                data.append(metatensor.torch.join(field, axis="samples", **join_kwargs))
            elif isinstance(field[0], torch.Tensor):  # torch.Tensor type
                data.append(torch.vstack(field))
            else:
                data.append(field)

        else:  # otherwise just keep as a list
            data.append(field)

    return namedtuple("Batch", names)(*data)
