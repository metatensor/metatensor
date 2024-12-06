from typing import Dict, List

from ._backend import (
    TensorBlock,
    TensorMap,
    is_metatensor_class,
    torch_jit_is_scripting,
    torch_jit_script,
)
from .slice import SliceSelection, _check_slice_args, _slice_block


def _split_block(
    block: TensorBlock,
    axis: str,
    selections: List[SliceSelection],
) -> List[TensorBlock]:
    """
    Splits a TensorBlock into multiple blocks, as in the public function
    :py:func:`split_block` but with no input checks.

    Note that the block is currently split into N new blocks by performing N number of
    slice operations. There may be a more efficient way of doing it, but this is not yet
    implemented.
    """
    new_blocks: List[TensorBlock] = []
    for selection in selections:
        # perform the slice either along the samples or properties axis
        new_block = _slice_block(block, axis=axis, selection=selection)
        new_blocks.append(new_block)

    return new_blocks


def _check_split_args(block: TensorBlock, axis: str, selections: List[SliceSelection]):
    """
    Checks the arguments passed to :py:func:`split` and :py:func:`split_block`.
    """
    # Check types
    if not torch_jit_is_scripting():
        if not isinstance(axis, str):
            raise TypeError(f"axis must be a string, not {type(axis)}")

        if not isinstance(selections, list):
            raise TypeError(f"`selections` must be a list, not {type(selections)}")

    if axis not in ["samples", "properties"]:
        raise ValueError("axis must be either 'samples' or 'properties'")

    # If passed as an empty list, return now
    if len(selections) == 0:
        return

    # Delegate to slice the checking of the selections
    for selection in selections:
        _check_slice_args(block, axis, selection)


@torch_jit_script
def split(
    tensor: TensorMap,
    axis: str,
    selections: List[SliceSelection],
) -> List[TensorMap]:
    """
    Split a :py:class:`TensorMap` into multiple :py:class:`TensorMap`.

    The operation is based on some specified groups of indices, along either the
    "samples" or "properties" ``axis``. The length of the returned list is equal to the
    number of :py:class:`Labels` objects passed in ``selections``. Each returned
    :py:class`TensorMap` will have the same keys and number of blocks at the input
    ``tensor``, but with the dimensions of the blocks reduced to only contain the
    specified indices for the corresponding group.

    For example, to split a tensor along the ``"samples"`` axis, according to the
    ``"system"`` index, where system 0, 6, and 7 are in the first returned
    :py:class`TensorMap`; 2, 3, and 4 in the second; and 1, 5, 8, 9, and 10 in the
    third:

    >>> import numpy as np
    >>> from metatensor import Labels, TensorBlock, TensorMap
    >>> import metatensor
    >>> block = TensorBlock(
    ...     values=np.random.rand(11, 3),
    ...     samples=Labels(
    ...         names=["system"],
    ...         values=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1),
    ...     ),
    ...     components=[],
    ...     properties=Labels.range("properties", 3),
    ... )
    >>> keys = Labels(names=["key"], values=np.array([[0]]))
    >>> tensor = TensorMap(keys, [block])
    >>> splitted = metatensor.split(
    ...     tensor,
    ...     axis="samples",
    ...     selections=[
    ...         Labels(names=["system"], values=np.array([[0], [6], [7]])),
    ...         Labels(names=["system"], values=np.array([[2], [3], [4]])),
    ...         Labels(names=["system"], values=np.array([[1], [5], [8], [10]])),
    ...     ],
    ... )
    >>> len(splitted)
    3
    >>> splitted[0].block(0).samples
    Labels(
        system
          0
          6
          7
    )
    >>> splitted[1].block(0).samples
    Labels(
        system
          2
          3
          4
    )
    >>> splitted[2].block(0).samples
    Labels(
        system
          1
          5
          8
          10
    )

    :param tensor: a :py:class:`TensorMap` to be split
    :param axis: a str, either "samples" or "properties", that indicates the
        :py:class:`TensorBlock` axis along which the named index (or indices) in
        ``selections`` belongs. Each :py:class:`TensorBlock` in each returned
        :py:class:`TensorMap` could have a reduced dimension along this axis, but the
        other axes will remain the same size.
    :param selections: a list of selections specifying what should be included in the
        different outputs. Each entry in the list can either be a :py:class:`Labels`
        selection, an array or a ``List[int]`` indicating the raw indices that should be
        kept. The list can mix different types of selections. When using
        :py:class:`Labels` selection, only a subset of the corresponding dimension names
        can be specified, and any entry with matching values will be selected.

    :return: a list of:py:class:`TensorMap` that corresponds to the split input
        ``tensor``. Each tensor in the returned list contains only the named indices in
        the respective py:class:`Labels` object of ``selections``.
    """
    # Check input args
    if not torch_jit_is_scripting():
        if not is_metatensor_class(tensor, TensorMap):
            raise TypeError(
                f"`tensor` must be a metatensor TensorMap, not {type(tensor)}"
            )

    _check_split_args(tensor.block(0), axis, selections)

    all_new_blocks: Dict[int, List[TensorBlock]] = {}
    for group_i in range(len(selections)):
        empty_list: List[TensorBlock] = []
        all_new_blocks[group_i] = empty_list

    for key_index in range(len(tensor.keys)):
        key = tensor.keys.entry(key_index)
        new_blocks = _split_block(tensor[key], axis, selections)

        for group_i, new_block in enumerate(new_blocks):
            all_new_blocks[group_i].append(new_block)

    return [
        TensorMap(keys=tensor.keys, blocks=all_new_blocks[group_i])
        for group_i in range(len(selections))
    ]


@torch_jit_script
def split_block(
    block: TensorBlock,
    axis: str,
    selections: List[SliceSelection],
) -> List[TensorBlock]:
    """
    Splits an input :py:class:`TensorBlock` into multiple :py:class:`TensorBlock`
    objects based on some specified ``selections``, along either the ``"samples"`` or
    ``"properties"`` ``axis``. The length of the returned list is equal to the number of
    selections passed in ``selections``. Each returned :py:class`TensorBlock` will have
    the same keys and number of blocks at the input ``tensor``, but with the dimensions
    of the blocks reduced to only contain the specified indices for the corresponding
    group.

    For example, to split a block along the ``"samples"`` axis, according to the
    ``"system"`` index, where system 0, 6, and 7 are in the first returned
    :py:class`TensorBlock`; 2, 3, and 4 in the second; and 1, 5, 8, 9, and 10 in the
    third:

    >>> import numpy as np
    >>> from metatensor import Labels, TensorBlock, TensorMap
    >>> import metatensor
    >>> block = TensorBlock(
    ...     values=np.random.rand(11, 3),
    ...     samples=Labels(
    ...         names=["system"],
    ...         values=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1),
    ...     ),
    ...     components=[],
    ...     properties=Labels.range("properties", 3),
    ... )
    >>> splitted = metatensor.split_block(
    ...     block,
    ...     axis="samples",
    ...     selections=[
    ...         Labels(names=["system"], values=np.array([[0], [6], [7]])),
    ...         Labels(names=["system"], values=np.array([[2], [3], [4]])),
    ...         Labels(names=["system"], values=np.array([[1], [5], [8], [10]])),
    ...     ],
    ... )
    >>> len(splitted)
    3
    >>> splitted[0].samples
    Labels(
        system
          0
          6
          7
    )
    >>> splitted[1].samples
    Labels(
        system
          2
          3
          4
    )
    >>> splitted[2].samples
    Labels(
        system
          1
          5
          8
          10
    )

    :param block: a :py:class:`TensorBlock` to be split
    :param axis: a str, either "samples" or "properties", that indicates the
        :py:class:`TensorBlock` axis along which the named index (or indices) in
        ``selections`` belongs. Each :py:class:`TensorBlock` returned could have a
        reduced dimension along this axis, but the other axes will remain the same size.
    :param selections: a list of selections specifying what should be included in the
        different outputs. Each entry in the list can either be a :py:class:`Labels`
        selection, an array or a ``List[int]`` indicating the raw indices that should be
        kept. The list can mix different types of selections. When using
        :py:class:`Labels` selection, only a subset of the corresponding dimension names
        can be specified, and any entry with matching values will be selected.

    :return: a list of:py:class:`TensorBlock` that corresponds to the split input
        ``block``. Each block in the returned list contains only the named indices in
        the respective py:class:`Labels` object of ``selections``.
    """
    # Check input args
    if not torch_jit_is_scripting():
        if not is_metatensor_class(block, TensorBlock):
            raise TypeError(
                f"`block` must be a metatensor TensorBlock, not {type(block)}"
            )

    _check_split_args(block, axis, selections)

    return _split_block(block, axis, selections)
