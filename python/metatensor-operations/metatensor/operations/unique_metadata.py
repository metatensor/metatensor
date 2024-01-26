"""
Module for finding unique metadata for TensorMaps and TensorBlocks
"""

from typing import List, Optional, Tuple, Union

from . import _dispatch
from ._classes import (
    Labels,
    TensorBlock,
    TensorMap,
    check_isinstance,
    torch_jit_is_scripting,
)


def unique_metadata(
    tensor: TensorMap,
    axis: str,
    names: Union[List[str], Tuple[str], str],
    gradient: Optional[str] = None,
) -> Labels:
    """
    Returns a :py:class:`Labels` object containing the unique metadata across
    all blocks of the input :py:class:`TensorMap`  ``tensor``. Unique Labels are
    returned for the specified ``axis`` (either ``"samples"`` or
    ``"properties"``) and metadata ``names``.

    Passing ``gradient`` as a ``str`` corresponding to a gradient parameter (for
    instance ``"cell"`` or ``"positions"``) returns the unique indices only for
    the gradient blocks. Note that gradient blocks by definition have the same
    properties metadata as their parent :py:class:`TensorBlock`.

    An empty :py:class:`Labels` object is returned if there are no indices in
    the (gradient) blocks of ``tensor`` corresponding to the specified ``axis``
    and ``names``. This will have length zero but the names will be the same as
    passed in ``names``.

    For example, to find the unique ``"structure"`` indices in the ``"samples"``
    metadata present in a given :py:class:`TensorMap`:

    .. code-block:: python

        import metatensor

        unique_structures = metatensor.unique_metadata(
            tensor,
            axis="samples",
            names=["structure"],
        )

    Or, to find the unique ``"atom"`` indices in the ``"samples"`` metadata
    present in the ``"positions"`` gradient blocks of a given
    :py:class:`TensorMap`:

    .. code-block:: python

        unique_grad_atoms = metatensor.unique_metadata(
            tensor,
            axis="samples",
            names=["atom"],
            gradient="positions",
        )

    The unique indices can then be used to split the :py:class:`TensorMap` into
    several smaller :py:class:`TensorMap` objects. Say, for example, that the
    ``unique_structures`` from the example above are:

    .. code-block:: python

        Labels(
            [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)],
            dtype=[("structure", "<i4")],
        )

    Then, the following code will split the :py:class:`TensorMap` into 2
    :py:class:`TensorMap` objects, with first containing structure indices 0-3
    and the second containing structure indices 4-9:

    .. code-block:: python

        import metatensor

        [tensor_1, tensor_2] = metatensor.split(
            tensor,
            axis="samples",
            grouped_labels=[unique_structures[:4], unique_structures[4:]],
        )

    :param tensor: the :py:class:`TensorMap` to find unique indices for.
    :param axis: a ``str``, either ``"samples"`` or ``"properties"``,
        corresponding to the ``axis`` along which the named unique indices
        should be found.
    :param names: a ``str``, ``list`` of ``str``, or ``tuple`` of ``str``
        corresponding to the name(s) of the indices along the specified ``axis``
        for which the unique values should be found.
    :param gradient: a ``str`` corresponding to the gradient parameter name for
        the gradient blocks to find the unique indices for. If :py:obj:`None`
        (default), the unique indices of the regular :py:class:`TensorBlock`
        objects will be calculated.

    :return: a sorted :py:class:`Labels` object containing the unique metadata
        for the blocks of the input ``tensor`` or its gradient blocks for the
        specified parameter. Each element in the returned :py:class:`Labels`
        object has len(``names``) entries.
    """
    # Parse input args
    if not torch_jit_is_scripting():
        if not check_isinstance(tensor, TensorMap):
            raise TypeError(
                f"`tensor` must be a metatensor TensorMap, not {type(tensor)}"
            )

    names = (
        [names]
        if isinstance(names, str)
        else (list(names) if isinstance(names, tuple) else names)
    )
    _check_args(axis, names, gradient)

    # Make a list of the blocks to find unique indices for
    if gradient is None:
        blocks = tensor.blocks()
    else:
        blocks = [block.gradient(gradient) for block in tensor.blocks()]

    return _unique_from_blocks(blocks, axis, names)


def unique_metadata_block(
    block: TensorBlock,
    axis: str,
    names: Union[List[str], Tuple[str], str],
    gradient: Optional[str] = None,
) -> Labels:
    """
    Returns a :py:class:`Labels` object containing the unique metadata in the
    input :py:class:`TensorBlock`  ``block``, for the specified ``axis`` (either
    ``"samples"`` or ``"properties"``) and metadata ``names``.

    Passing ``gradient`` as a ``str`` corresponding to a gradient parameter (for
    instance ``"cell"`` or ``"positions"``) returns the unique indices only for
    the gradient block associated with ``block``. Note that gradient blocks by
    definition have the same properties metadata as their parent
    :py:class:`TensorBlock`.

    An empty :py:class:`Labels` object is returned if there are no indices in
    the (gradient) blocks of ``tensor`` corresponding to the specified ``axis``
    and ``names``. This will have length zero but the names will be the same as
    passed in ``names``.

    For example, to find the unique ``"structure"`` indices in the ``"samples"``
    metadata present in a given :py:class:`TensorBlock`:

    .. code-block:: python

        import metatensor

        unique_samples = metatensor.unique_metadata_block(
            block,
            axis="samples",
            names=["structure"],
        )

    To find the unique ``"atom"`` indices along the ``"samples"`` axis present
    in the ``"positions"`` gradient block of a given :py:class:`TensorBlock`:

    .. code-block:: python

        unique_grad_samples = metatensor.unique_metadata_block(
            block,
            axis="samples",
            names=["atom"],
            gradient="positions",
        )

    :param block: the :py:class:`TensorBlock` to find unique indices for.
    :param axis: a str, either ``"samples"`` or ``"properties"``, corresponding
        to the ``axis`` along which the named unique metadata should be found.
    :param names: a ``str``, ``list`` of ``str``, or ``tuple`` of ``str``
        corresponding to the name(s) of the metadata along the specified
        ``axis`` for which the unique indices should be found.
    :param gradient: a ``str`` corresponding to the gradient parameter name for
        the gradient blocks to find the unique metadata for. If :py:obj:`None`
        (default), the unique metadata of the regular :py:class:`TensorBlock`
        objects will be calculated.

    :return: a sorted :py:class:`Labels` object containing the unique metadata
        for the input ``block`` or its gradient for the specified parameter.
        Each element in the returned :py:class:`Labels` object has
        len(``names``) entries.
    """
    # Parse input args
    if not torch_jit_is_scripting():
        if not check_isinstance(block, TensorBlock):
            raise TypeError(
                f"`block` must be a metatensor TensorBlock, not {type(block)}"
            )

    names = (
        [names]
        if isinstance(names, str)
        else (list(names) if isinstance(names, tuple) else names)
    )
    _check_args(axis, names, gradient)

    # Make a list of the blocks to find unique indices for
    if gradient is None:
        blocks = [block]
    else:
        blocks = [block.gradient(gradient)]

    return _unique_from_blocks(blocks, axis, names)


def _unique_from_blocks(
    blocks: List[TensorBlock],
    axis: str,
    names: List[str],
) -> Labels:
    """
    Finds the unique metadata of a list of blocks along the given ``axis`` and
    for the specified ``names``.
    """
    all_values = []
    for block in blocks:
        if axis == "samples":
            all_values.append(block.samples.view(names).values)
        else:
            assert axis == "properties"
            all_values.append(block.properties.view(names).values)

    unique_values = _dispatch.unique(_dispatch.concatenate(all_values, axis=0), axis=0)
    return Labels(names=names, values=unique_values)


def _check_args(
    axis: str,
    names: List[str],
    gradient: Optional[str] = None,
):
    """Checks input args for `unique_metadata_block`"""

    if not torch_jit_is_scripting():
        if not isinstance(axis, str):
            raise TypeError(f"`axis` must be a string, not {type(axis)}")

        if not isinstance(names, list):
            raise TypeError(f"`names` must be a list of strings, not {type(names)}")

        for name in names:
            if not isinstance(name, str):
                raise TypeError(f"`names` elements must be a strings, not {type(name)}")

    if gradient is not None:
        if not torch_jit_is_scripting():
            if not isinstance(gradient, str):
                raise TypeError(f"`gradient` must be a string, not {type(gradient)}")

    if axis not in ["samples", "properties"]:
        raise ValueError(
            f"`axis` must be either 'samples' or 'properties', not '{axis}'"
        )
