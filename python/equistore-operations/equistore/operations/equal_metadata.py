"""
Module for checking equivalence in metadata between 2 TensorMaps
"""
from typing import List, Optional

from equistore.core import TensorBlock, TensorMap

from ._utils import (
    NotEqualError,
    _check_blocks,
    _check_same_gradients,
    _check_same_keys,
)


def equal_metadata(
    tensor_1: TensorMap, tensor_2: TensorMap, check: Optional[List] = None
) -> bool:
    """
    Checks if two :py:class:`TensorMap` objects have the same metadata,
    returning a bool.

    The equivalence of the keys of the two :py:class:`TensorMap` objects is
    always checked. If ``check`` is none (the default), all metadata (i.e. the
    samples, components, and properties of each block) is checked to contain the
    same values in the same order.

    Passing ``check`` as a list of strings will only check the metadata specified.
    Allowed values to pass are "samples", "components", and "properties".

    :param tensor_1: The first :py:class:`TensorMap`.
    :param tensor_2: The second :py:class:`TensorMap` to compare to the first.
    :param check: A list of strings specifying which metadata of each block to
        check. If none, all metadata is checked. Allowed values are "samples",
        "components", and "properties".

    :return: True if the metadata of the two :py:class:`TensorMap` objects are
        equal, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> import equistore
    >>> from equistore import Labels, TensorBlock, TensorMap
    >>> tensor_1 = TensorMap(
    ...     keys=Labels(
    ...         names=["key_1", "key_2"],
    ...         values=np.array([[1, 0], [2, 2]]),
    ...     ),
    ...     blocks=[
    ...         TensorBlock(
    ...             values=np.full((4, 3, 1), 4.0),
    ...             samples=Labels(["samples"], np.array([[0], [1], [4], [5]])),
    ...             components=[Labels.arange("components", 3)],
    ...             properties=Labels(["p_1", "p_2"], np.array([[0, 1]])),
    ...         ),
    ...         TensorBlock(
    ...             values=np.full((4, 3, 1), 4.0),
    ...             samples=Labels(["samples"], np.array([[0], [1], [4], [5]])),
    ...             components=[Labels.arange("components", 3)],
    ...             properties=Labels(["p_1", "p_2"], np.array([[0, 1]])),
    ...         ),
    ...     ],
    ... )
    >>> tensor_2 = TensorMap(
    ...     keys=Labels(
    ...         names=["key_1", "key_2"],
    ...         values=np.array([[1, 0], [2, 2]]),
    ...     ),
    ...     blocks=[
    ...         TensorBlock(
    ...             values=np.full((4, 3, 1), 4.0),
    ...             samples=Labels(["samples"], np.array([[0], [1], [4], [5]])),
    ...             components=[Labels.arange("components", 3)],
    ...             properties=Labels(["p_3", "p_4"], np.array([[0, 1]])),
    ...         ),
    ...         TensorBlock(
    ...             values=np.full((4, 3, 1), 4.0),
    ...             samples=Labels(["samples"], np.array([[0], [1], [4], [5]])),
    ...             components=[Labels.arange("components", 3)],
    ...             properties=Labels(["p_3", "p_4"], np.array([[0, 1]])),
    ...         ),
    ...     ],
    ... )
    >>> equistore.equal_metadata(tensor_1, tensor_2)
    False
    >>> equistore.equal_metadata(
    ...     tensor_1,
    ...     tensor_2,
    ...     check=["samples", "components"],
    ... )
    True
    """
    # Check input args
    if not isinstance(tensor_1, TensorMap):
        raise TypeError(f"`tensor_1` must be a TensorMap, not {type(tensor_1)}")
    if not isinstance(tensor_2, TensorMap):
        raise TypeError(f"`tensor_2` must be a TensorMap, not {type(tensor_2)}")
    if not isinstance(check, (list, type(None))):
        raise TypeError(f"`check` must be a list, not {type(check)}")
    if check is None:
        check = ["samples", "components", "properties"]
    for metadata in check:
        if not isinstance(metadata, str):
            raise TypeError(
                f"`check` must be a list of strings, got list of {type(metadata)}"
            )
        if metadata not in ["samples", "components", "properties"]:
            raise ValueError(f"Invalid metadata to check: {metadata}")
    # Check equivalence in keys
    try:
        _check_same_keys(tensor_1, tensor_2, "equal_metadata")
    except NotEqualError:
        return False

    # Loop over the blocks
    for key in tensor_1.keys:
        block_1 = tensor_1[key]
        block_2 = tensor_2[key]

        # Check metadata of the blocks
        try:
            _check_blocks(block_1, block_2, check, "equal_metadata")
        except NotEqualError:
            return False

        # Check metadata of the gradients
        try:
            _check_same_gradients(block_1, block_2, check, "equal_metadata")
        except NotEqualError:
            return False
    return True


def equal_metadata_block(
    block_1: TensorBlock, block_2: TensorBlock, check: Optional[List] = None
) -> bool:
    """
    Checks if two :py:class:`TensorBlock` objects have the same metadata,
    returning a bool.

    If ``check`` is none (the default), all metadata (i.e. the samples,
    components, and properties of each block) is checked to contain the same
    values in the same order.

    Passing ``check`` as a list of strings will only check the metadata specified.
    Allowed values to pass are "samples", "components", and "properties".

    :param block_1: The first :py:class:`TensorBlock`.
    :param block_2: The second :py:class:`TensorBlock` to compare to the first.
    :param check: A list of strings specifying which metadata of each block to
        check. If none, all metadata is checked. Allowed values are "samples",
        "components", and "properties".

    :return: True if the metadata of the two :py:class:`TensorBlock` objects are
        equal, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> import equistore
    >>> from equistore import Labels, TensorBlock
    >>> block_1 = TensorBlock(
    ...     values=np.full((4, 3, 1), 4.0),
    ...     samples=Labels(["samples"], np.array([[0], [1], [4], [5]])),
    ...     components=[Labels.arange("components", 3)],
    ...     properties=Labels(["p_1", "p_2"], np.array([[0, 1]])),
    ... )
    >>> block_2 = TensorBlock(
    ...     values=np.full((4, 3, 1), 4.0),
    ...     samples=Labels(["samples"], np.array([[0], [1], [4], [5]])),
    ...     components=[Labels.arange("components", 3)],
    ...     properties=Labels(["p_3", "p_4"], np.array([[0, 1]])),
    ... )
    >>> equal_metadata_block(block_1, block_2)
    False
    >>> equal_metadata_block(
    ...     block_1,
    ...     block_2,
    ...     check=["samples", "components"],
    ... )
    True
    """
    # Check input args
    if not isinstance(block_1, TensorBlock):
        raise TypeError(f"`block_1` must be a TensorBlock, not {type(block_1)}")
    if not isinstance(block_2, TensorBlock):
        raise TypeError(f"`block_2` must be a TensorBlock, not {type(block_2)}")
    if not isinstance(check, (list, type(None))):
        raise TypeError(f"`check` must be a list, not {type(check)}")
    if check is None:
        check = ["samples", "components", "properties"]
    for metadata in check:
        if not isinstance(metadata, str):
            raise TypeError(
                f"`check` must be a list of strings, got list of {type(metadata)}"
            )
        if metadata not in ["samples", "components", "properties"]:
            raise ValueError(f"Invalid metadata to check: {metadata}")

    # Check metadata of the blocks
    try:
        _check_blocks(block_1, block_2, check, "equal_metadata_block")
    except NotEqualError:
        return False

    # Check metadata of the gradients
    try:
        _check_same_gradients(block_1, block_2, check, "equal_metadata_block")
    except NotEqualError:
        return False
    return True
