"""
Module for checking equivalence in metadata between 2 TensorMaps
"""
from typing import List, Optional

from equistore.core import TensorBlock, TensorMap

from ._utils import (
    NotEqualError,
    _check_blocks_impl,
    _check_same_gradients_impl,
    _check_same_keys_impl,
)


def _equal_metadata_impl(
    tensor_1: TensorMap, tensor_2: TensorMap, check: Optional[List] = None
) -> str:
    if not isinstance(tensor_1, TensorMap):
        return f"`tensor_1` must be a TensorMap, not {type(tensor_1)}"
    if not isinstance(tensor_2, TensorMap):
        return f"`tensor_2` must be a TensorMap, not {type(tensor_2)}"
    if not isinstance(check, (list, type(None))):
        return f"`check` must be a list, not {type(check)}"
    if check is None:
        check = ["samples", "components", "properties"]
    for metadata in check:
        if not isinstance(metadata, str):
            return f"`check` must be a list of strings, got list of {type(metadata)}"

        if metadata not in ["samples", "components", "properties"]:
            return f"Invalid metadata to check: {metadata}"

    message = _check_same_keys_impl(tensor_1, tensor_2, "equal_metadata_raise")
    if message != "":
        return message

    for key in tensor_1.keys:
        message = _equal_metadata_block_impl(tensor_1[key], tensor_2[key], check=check)
        if message != "":
            return message

    return ""


def _equal_metadata_block_impl(
    block_1: TensorBlock, block_2: TensorBlock, check: Optional[List] = None
) -> str:
    if not isinstance(block_1, TensorBlock):
        return f"`block_1` must be a TensorBlock, not {type(block_1)}"
    if not isinstance(block_2, TensorBlock):
        return f"`block_2` must be a TensorBlock, not {type(block_2)}"
    if not isinstance(check, (list, type(None))):
        return f"`check` must be a list, not {type(check)}"
    if check is None:
        check = ["samples", "components", "properties"]
    for metadata in check:
        if not isinstance(metadata, str):
            return f"`check` must be a list of strings, got list of {type(metadata)}"
        if metadata not in ["samples", "components", "properties"]:
            return f"Invalid metadata to check: {metadata}"

    check_blocks_message = _check_blocks_impl(
        block_1, block_2, check, "equal_metadata_block_raise"
    )
    if check_blocks_message != "":
        return check_blocks_message
    check_same_gradient_message = _check_same_gradients_impl(
        block_1, block_2, check, "equal_metadata_block_raise"
    )
    if check_same_gradient_message != "":
        return check_same_gradient_message

    return ""


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
    ...             components=[Labels.range("components", 3)],
    ...             properties=Labels(["p_1", "p_2"], np.array([[0, 1]])),
    ...         ),
    ...         TensorBlock(
    ...             values=np.full((4, 3, 1), 4.0),
    ...             samples=Labels(["samples"], np.array([[0], [1], [4], [5]])),
    ...             components=[Labels.range("components", 3)],
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
    ...             components=[Labels.range("components", 3)],
    ...             properties=Labels(["p_3", "p_4"], np.array([[0, 1]])),
    ...         ),
    ...         TensorBlock(
    ...             values=np.full((4, 3, 1), 4.0),
    ...             samples=Labels(["samples"], np.array([[0], [1], [4], [5]])),
    ...             components=[Labels.range("components", 3)],
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
    return not bool(_equal_metadata_impl(tensor_1, tensor_2, check))


def equal_metadata_raise(
    tensor_1: TensorMap, tensor_2: TensorMap, check: Optional[List] = None
):
    """
    Raise a :py:class:`NotEqualError` if two :py:class:`TensorMap` have unequal
    metadata.

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
    :raises NotEqualError: If the metadata is not the same.

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
    ...             components=[Labels.range("components", 3)],
    ...             properties=Labels(["p_1", "p_2"], np.array([[0, 1]])),
    ...         ),
    ...         TensorBlock(
    ...             values=np.full((4, 3, 1), 4.0),
    ...             samples=Labels(["samples"], np.array([[0], [1], [4], [5]])),
    ...             components=[Labels.range("components", 3)],
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
    ...             components=[Labels.range("components", 3)],
    ...             properties=Labels(["p_3", "p_4"], np.array([[0, 1]])),
    ...         ),
    ...         TensorBlock(
    ...             values=np.full((4, 3, 1), 4.0),
    ...             samples=Labels(["samples"], np.array([[0], [1], [4], [5]])),
    ...             components=[Labels.range("components", 3)],
    ...             properties=Labels(["p_3", "p_4"], np.array([[0, 1]])),
    ...         ),
    ...     ],
    ... )
    >>> equistore.equal_metadata_raise(tensor_1, tensor_2)
    Traceback (most recent call last):
        ...
    equistore.operations._utils.NotEqualError: inputs to 'equal_metadata_block_raise' should have the same properties:
    properties names are not the same or not in the same order
    >>> equistore.equal_metadata_raise(
    ...     tensor_1,
    ...     tensor_2,
    ...     check=["samples", "components"],
    ... )
    """  # noqa: E501
    message = _equal_metadata_impl(tensor_1, tensor_2, check)
    if message != "":
        raise NotEqualError(message)


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
    ...     components=[Labels.range("components", 3)],
    ...     properties=Labels(["p_1", "p_2"], np.array([[0, 1]])),
    ... )
    >>> block_2 = TensorBlock(
    ...     values=np.full((4, 3, 1), 4.0),
    ...     samples=Labels(["samples"], np.array([[0], [1], [4], [5]])),
    ...     components=[Labels.range("components", 3)],
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
    return not bool(_equal_metadata_block_impl(block_1, block_2, check))


def equal_metadata_block_raise(
    block_1: TensorBlock, block_2: TensorBlock, check: Optional[List] = None
):
    """
    Raise a :py:class:`NotEqualError` if two :py:class:`TensorBlock` have unequal
    metadata.

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
    :raises NotEqualError: If the metadata is not the same.

    Examples
    --------
    >>> import numpy as np
    >>> import equistore
    >>> from equistore import Labels, TensorBlock
    >>> block_1 = TensorBlock(
    ...     values=np.full((4, 3, 1), 4.0),
    ...     samples=Labels(["samples"], np.array([[0], [1], [4], [5]])),
    ...     components=[Labels.range("components", 3)],
    ...     properties=Labels(["p_1", "p_2"], np.array([[0, 1]])),
    ... )
    >>> block_2 = TensorBlock(
    ...     values=np.full((4, 3, 1), 4.0),
    ...     samples=Labels(["samples"], np.array([[0], [1], [4], [5]])),
    ...     components=[Labels.range("components", 3)],
    ...     properties=Labels(["p_3", "p_4"], np.array([[0, 1]])),
    ... )
    >>> equistore.equal_metadata_block_raise(block_1, block_2)
    Traceback (most recent call last):
        ...
    equistore.operations._utils.NotEqualError: inputs to 'equal_metadata_block_raise' should have the same properties:
    properties names are not the same or not in the same order
    >>> equistore.equal_metadata_block_raise(
    ...     block_1, block_2, check=["samples", "components"]
    ... )
    """  # noqa: E501
    message = _equal_metadata_block_impl(block_1, block_2, check)
    if message != "":
        raise NotEqualError(message)
