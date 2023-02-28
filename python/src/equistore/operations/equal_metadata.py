"""
Module for checking equivalence in metadata between 2 TensorMaps
"""
from typing import List, Optional

from ..tensor import TensorMap
from ._utils import _check_blocks, _check_maps, _check_same_gradients


def equal_metadata(
    tensor_1: TensorMap, tensor_2: TensorMap, check: Optional[List] = None
) -> bool:
    """
    Checks if two :py:class:`TensorMap` objects have the same metadata,
    returning a bool.

    Equivalence in the keys of the two :py:class:`TensorMap` objects is always
    checked. In addition to this, users can control the metadata of the blocks
    that is checked. If `check` is none (the default), all metadata is checked
    for exact equivalence in values and order, i.e. the samples, components, and
    properties of each block.

    Passing `check` as a list of strings will only check the metadata specified.
    Allowed values to pass are "samples", "components", and "properties".

    :param tensor_1: The first :py:class:`TensorMap`.
    :param tensor_2: The second :py:class:`TensorMap` to compare to the first.
    :param check: A list of strings specifying which metadata of each block to
        check. If none, all metadata is checked. Allowed values are "samples",
        "components", and "properties".

    Examples
    --------
    >>> import numpy as np
    >>> from equistore import Labels, TensorBlock, TensorMap
    >>> from equistore.operations import equal_metadata
    >>> tensor_1 = TensorMap(
    ...     keys=Labels(
    ...         names=["key_1", "key_2"],
    ...         values=np.array([[1, 0], [2, 2]]),
    ...     ),
    ...     blocks=[
    ...         TensorBlock(
    ...             values=np.full((4, 3, 1), 4.0),
    ...             samples=Labels(
    ...                 ["samples"],
    ...                 np.array([[0], [1], [4], [5]])
    ...             ),
    ...             components=[
    ...                 Labels(
    ...                     ["components"],
    ...                     np.array([[0], [1], [2]])
    ...                 )
    ...             ],
    ...             properties=Labels(
    ...                 ["p_1", "p_2"],
    ...                 np.array([[0, 1]])
    ...             ),
    ...         ),
    ...         TensorBlock(
    ...             values=np.full((4, 3, 1), 4.0),
    ...             samples=Labels(
    ...                 ["samples"],
    ...                 np.array([[0], [1], [4], [5]])
    ...             ),
    ...             components=[
    ...                 Labels(
    ...                     ["components"],
    ...                     np.array([[0], [1], [2]])
    ...                 )
    ...             ],
    ...             properties=Labels(
    ...                 ["p_1", "p_2"],
    ...                 np.array([[0, 1]])
    ...             ),
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
    ...             samples=Labels(
    ...                 ["samples"],
    ...                 np.array([[0], [1], [4], [5]])),
    ...             components=[
    ...                 Labels(
    ...                     ["components"],
    ...                     np.array([[0], [1], [2]])
    ...                 )
    ...             ],
    ...             properties=Labels(
    ...                 ["p_3", "p_4"],
    ...                 np.array([[0, 1]])
    ...             ),
    ...         ),
    ...         TensorBlock(
    ...             values=np.full((4, 3, 1), 4.0),
    ...             samples=Labels(
    ...                 ["samples"],
    ...                 np.array([[0], [1], [4], [5]])
    ...             ),
    ...             components=[
    ...                 Labels(
    ...                     ["components"],
    ...                     np.array([[0], [1], [2]])
    ...                 )
    ...             ],
    ...             properties=Labels(
    ...                 ["p_3", "p_4"],
    ...                 np.array([[0, 1]])
    ...             ),
    ...         ),
    ...     ],
    ... )
    >>> equal_metadata(tensor_1, tensor_2)
    False
    >>> equal_metadata(
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
        if metadata not in ["keys", "samples", "components", "properties"]:
            raise ValueError(f"Invalid metadata to check: {metadata}")
    # Check equivalence in keys
    try:
        _check_maps(tensor_1, tensor_2, "equal_metadata")
    except ValueError:
        return False

    # Loop over the blocks
    for key in tensor_1.keys:
        block_1 = tensor_1[key]
        block_2 = tensor_2[key]

        # Check metatdata of the blocks
        try:
            _check_blocks(block_1, block_2, check, "equal_metadata")
        except ValueError:
            return False

        # Check metadata of the gradients
        try:
            _check_same_gradients(block_1, block_2, check, "equal_metadata")
        except ValueError:
            return False
    return True
