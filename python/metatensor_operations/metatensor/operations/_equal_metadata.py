from typing import List, Union

from ._backend import (
    TensorBlock,
    TensorMap,
    isinstance_metatensor,
    torch_jit_is_scripting,
    torch_jit_script,
)
from ._utils import (
    NotEqualError,
    _check_blocks_impl,
    _check_same_gradients_impl,
    _check_same_keys_impl,
)


def _equal_metadata_impl(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    check: Union[List[str], str] = "all",
    check_gradients: bool = True,
) -> str:
    if not torch_jit_is_scripting():
        if not isinstance_metatensor(tensor_1, "TensorMap"):
            return f"`tensor_1` must be a metatensor TensorMap, not {type(tensor_1)}"
        if not isinstance_metatensor(tensor_2, "TensorMap"):
            return f"`tensor_2` must be a metatensor TensorMap, not {type(tensor_2)}"

    message = _check_same_keys_impl(tensor_1, tensor_2, "equal_metadata_raise")
    if message != "":
        return message

    for key in [tensor_1.keys[i] for i in range(len(tensor_1.keys))]:
        message = _equal_metadata_block_impl(
            tensor_1[key], tensor_2[key], check=check, check_gradients=check_gradients
        )
        if message != "":
            return message

    return ""


def _equal_metadata_block_impl(
    block_1: TensorBlock,
    block_2: TensorBlock,
    check: Union[List[str], str] = "all",
    check_gradients: bool = True,
) -> str:
    if not torch_jit_is_scripting():
        if not isinstance_metatensor(block_1, "TensorBlock"):
            return f"`block_1` must be a metatensor TensorBlock, not {type(block_1)}"
        if not isinstance_metatensor(block_2, "TensorBlock"):
            return f"`block_2` must be a metatensor TensorBlock, not {type(block_2)}"

    check_blocks_message = _check_blocks_impl(
        block_1,
        block_2,
        "equal_metadata_block_raise",
        check=check,
    )

    if check_blocks_message != "":
        return check_blocks_message

    if check_gradients:
        check_same_gradient_message = _check_same_gradients_impl(
            block_1,
            block_2,
            "equal_metadata_block_raise",
            check=check,
        )

        if check_same_gradient_message != "":
            return check_same_gradient_message

    return ""


@torch_jit_script
def equal_metadata(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    check: Union[List[str], str] = "all",
    check_gradients: bool = True,
) -> bool:
    """
    Checks if two :py:class:`TensorMap` objects have the same metadata, returning a
    bool.

    The equivalence of the keys of the two :py:class:`TensorMap` objects is always
    checked.

    If ``check`` is ``'all'`` (the default), all metadata (i.e. the samples, components,
    and properties of each block) is checked to contain the same values in the same
    order. Passing ``check`` as a list of strings will only check the specified
    metadata. Allowed values to pass are ``'samples'``, ``'components'``, and
    ``'properties'``.

    :param tensor_1: The first :py:class:`TensorMap`.
    :param tensor_2: The second :py:class:`TensorMap` to compare to the first.
    :param check: Which parts of the metadata to check. This can be a list containing
        any of ``'samples'``, ``'components'``, and ``'properties'``; or the string
        ``'all'`` to check everything. Defaults to ``'all'``.
    :param check_gradients: Whether to check if the gradients' metadata is also equal.
        If `True`, `check` is also used to determine which metadata to check for the
        gradients.

    :return: True if the metadata of the two :py:class:`TensorMap` objects are equal,
        False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> import metatensor as mts
    >>> from metatensor import Labels, TensorBlock, TensorMap
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
    >>> mts.equal_metadata(tensor_1, tensor_2)
    False
    >>> mts.equal_metadata(
    ...     tensor_1,
    ...     tensor_2,
    ...     check=("samples", "components"),
    ... )
    True
    """
    return not bool(_equal_metadata_impl(tensor_1, tensor_2, check, check_gradients))


@torch_jit_script
def equal_metadata_raise(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    check: Union[List[str], str] = "all",
    check_gradients: bool = True,
):
    """
    Raise a :py:class:`NotEqualError` if two :py:class:`TensorMap` have unequal
    metadata.

    The equivalence of the keys of the two :py:class:`TensorMap` objects is always
    checked.

    If ``check`` is ``'all'`` (the default), all metadata (i.e. the samples, components,
    and properties of each block) is checked to contain the same values in the same
    order. Passing ``check`` as a list of strings will only check the specified
    metadata. Allowed values to pass are ``'samples'``, ``'components'``, and
    ``'properties'``.

    :param tensor_1: The first :py:class:`TensorMap`.
    :param tensor_2: The second :py:class:`TensorMap` to compare to the first.
    :param check: Which parts of the metadata to check. This can be a list containing
        any of ``'samples'``, ``'components'``, and ``'properties'``; or the string
        ``'all'`` to check everything. Defaults to ``'all'``.
    :param check_gradients: Whether to check if the gradients' metadata is also equal.
        If `True`, `check` is also used to determine which metadata to check for the
        gradients.
    :raises NotEqualError: If the metadata is not the same.

    Examples
    --------
    >>> import numpy as np
    >>> import metatensor as mts
    >>> from metatensor import Labels, TensorBlock, TensorMap
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
    >>> mts.equal_metadata_raise(tensor_1, tensor_2)
    Traceback (most recent call last):
        ...
    metatensor.operations._utils.NotEqualError: inputs to 'equal_metadata_block_raise' \
should have the same properties, but they are not the same or not in the same order
    >>> mts.equal_metadata_raise(
    ...     tensor_1,
    ...     tensor_2,
    ...     check=("samples", "components"),
    ... )
    """
    message = _equal_metadata_impl(tensor_1, tensor_2, check, check_gradients)
    if message != "":
        raise NotEqualError(message)


@torch_jit_script
def equal_metadata_block(
    block_1: TensorBlock,
    block_2: TensorBlock,
    check: Union[List[str], str] = "all",
    check_gradients: bool = True,
) -> bool:
    """
    Checks if two :py:class:`TensorBlock` objects have the same metadata, returning a
    bool.

    If ``check`` is ``'all'`` (the default), all metadata (i.e. the samples, components,
    and properties of each block) is checked to contain the same values in the same
    order. Passing ``check`` as a list of strings will only check the specified
    metadata. Allowed values to pass are ``'samples'``, ``'components'``, and
    ``'properties'``.

    :param block_1: The first :py:class:`TensorBlock`.
    :param block_2: The second :py:class:`TensorBlock` to compare to the first.
    :param check: Which parts of the metadata to check. This can be a list containing
        any of ``'samples'``, ``'components'``, and ``'properties'``; or the string
        ``'all'`` to check everything. Defaults to ``'all'``.
    :param check_gradients: Whether to check if the gradients' metadata is also equal.
        If `True`, `check` is also used to determine which metadata to check for the
        gradients.

    :return: True if the metadata of the two :py:class:`TensorBlock` objects are equal,
        False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> import metatensor as mts
    >>> from metatensor import Labels, TensorBlock
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
    ...     check=("samples", "components"),
    ... )
    True
    """
    return not bool(
        _equal_metadata_block_impl(block_1, block_2, check, check_gradients)
    )


@torch_jit_script
def equal_metadata_block_raise(
    block_1: TensorBlock,
    block_2: TensorBlock,
    check: Union[List[str], str] = "all",
    check_gradients: bool = True,
):
    """
    Raise a :py:class:`NotEqualError` if two :py:class:`TensorBlock` have unequal
    metadata.

    If ``check`` is ``'all'`` (the default), all metadata (i.e. the samples, components,
    and properties of each block) is checked to contain the same values in the same
    order. Passing ``check`` as a list of strings will only check the specified
    metadata. Allowed values to pass are ``'samples'``, ``'components'``, and
    ``'properties'``.

    :param block_1: The first :py:class:`TensorBlock`.
    :param block_2: The second :py:class:`TensorBlock` to compare to the first.
    :param check: A sequence of strings specifying which metadata of each block to
        check. If none, all metadata is checked. Allowed values are "samples",
        "components", and "properties".
    :param check_gradients: Whether to check if the gradients' metadata is also equal.
        If `True`, `check` is also used to determine which metadata to check for the
        gradients.
    :raises NotEqualError: If the metadata is not the same.

    Examples
    --------
    >>> import numpy as np
    >>> import metatensor as mts
    >>> from metatensor import Labels, TensorBlock
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
    >>> mts.equal_metadata_block_raise(block_1, block_2)
    Traceback (most recent call last):
        ...
    metatensor.operations._utils.NotEqualError: inputs to 'equal_metadata_block_raise' \
should have the same properties, but they are not the same or not in the same order
    >>> mts.equal_metadata_block_raise(
    ...     block_1, block_2, check=("samples", "components")
    ... )
    """
    message = _equal_metadata_block_impl(block_1, block_2, check, check_gradients)
    if message != "":
        raise NotEqualError(message)
