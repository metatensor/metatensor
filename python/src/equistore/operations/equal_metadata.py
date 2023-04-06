"""
Module for checking equivalence in metadata between 2 TensorMaps
"""
from typing import List, Optional

import numpy as np

from ..block import TensorBlock
from ..labels import Labels
from ..tensor import TensorMap


def equal_metadata(
    tensor_1: TensorMap, tensor_2: TensorMap, check: Optional[List] = None
) -> bool:
    """
    Checks if two :py:class:`TensorMap` objects have the same metadata,
    returning a bool.

    The equivalence of the keys of the two :py:class:`TensorMap` objects is
    always checked. If `check` is none (the default), all metadata (i.e. the
    samples, components, and properties of each block) is checked to contain the
    same values in the same order.

    Passing `check` as a list of strings will only check the metadata specified.
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
        if metadata not in ["samples", "components", "properties"]:
            raise ValueError(f"Invalid metadata to check: {metadata}")
    # Check equivalence in keys
    try:
        _check_same_keys(tensor_1, tensor_2, "equal_metadata")
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


def _check_same_keys(a: TensorMap, b: TensorMap, fname: str):
    """Check if metadata between two TensorMaps is consistent for an operation.

    The functions verifies that

    1. The key names are the same.
    2. The number of blocks in the same
    3. The block key indices are the same.

    :param a: first :py:class:`TensorMap` for check
    :param b: second :py:class:`TensorMap` for check
    """

    keys_a = a.keys
    keys_b = b.keys

    if keys_a.names != keys_b.names:
        raise ValueError(
            f"inputs to {fname} should have the same keys names, "
            f"got '{keys_a.names}' and '{keys_b.names}'"
        )

    if len(keys_a) != len(keys_b):
        raise ValueError(
            f"inputs to {fname} should have the same number of blocks, "
            f"got {len(keys_a)} and {len(keys_b)}"
        )

    if not np.all([key in keys_a for key in keys_b]):
        raise ValueError(f"inputs to {fname} should have the same keys")


def _check_blocks(a: TensorBlock, b: TensorBlock, props: List[str], fname: str):
    """Check if metadata between two TensorBlocks is consistent for an operation.

    The functions verifies that that the metadata of the given props is the same
    (length and indices).

    :param a: first :py:class:`TensorBlock` for check
    :param b: second :py:class:`TensorBlock` for check
    :param props: A list of strings containing the property to check.
                 Allowed values are ``'properties'`` or ``'samples'``,
                 ``'components'`` and ``'gradients'``.
    """
    for prop in props:
        err_msg = f"inputs to '{fname}' should have the same {prop}:\n"
        err_msg_len = f"{prop} of the two `TensorBlock` have different lengths"
        err_msg_1 = f"{prop} are not the same or not in the same order"
        err_msg_names = f"{prop} names are not the same or not in the same order"
        if prop == "samples":
            if not len(a.samples) == len(b.samples):
                raise ValueError(err_msg + err_msg_len)
            if not a.samples.names == b.samples.names:
                raise ValueError(err_msg + err_msg_names)
            if not np.all(a.samples == b.samples):
                raise ValueError(err_msg + err_msg_1)
        elif prop == "properties":
            if not len(a.properties) == len(b.properties):
                raise ValueError(err_msg + err_msg_len)
            if not a.properties.names == b.properties.names:
                raise ValueError(err_msg + err_msg_names)
            if not np.all(a.properties == b.properties):
                raise ValueError(err_msg + err_msg_1)
        elif prop == "components":
            if len(a.components) != len(b.components):
                raise ValueError(err_msg + err_msg_len)

            for c1, c2 in zip(a.components, b.components):
                if not (c1.names == c2.names):
                    raise ValueError(err_msg + err_msg_names)

                if not (len(c1) == len(c2)):
                    raise ValueError(err_msg + err_msg_1)

                if not np.all(c1 == c2):
                    raise ValueError(err_msg + err_msg_1)

        else:
            raise ValueError(
                f"{prop} is not a valid property to check, "
                "choose from ['samples', 'properties', 'components']"
            )


def _check_same_gradients(a: TensorBlock, b: TensorBlock, props: List[str], fname: str):
    """
    Check if metadata between two gradients's TensorBlocks is consistent
     for an operation.

    The functions verifies that the metadata of the given props is the same
    (length, names, values) and in the same order.

    If props is None it only checks if the ``'parameters'`` are consistent.

    :param a: first :py:class:`TensorBlock` for check
    :param b: second :py:class:`TensorBlock` for check
    :param props: A list of strings containing the property to check. Allowed
                 values are ``'samples'`` or ``'properties'``, ``'components'``.
                 To check only if the ``'parameters'`` are consistent use
                 ``props=None``, using ``'parameters'`` is allowed
                  even though deprecated.
    """
    err_msg = f"inputs to {fname} should have the same gradient:\n"
    grads_a = a.gradients_list()
    grads_b = b.gradients_list()

    if len(grads_a) != len(grads_b) or (
        not np.all([parameter in grads_b for parameter in grads_a])
    ):
        raise ValueError(f"inputs to {fname} should have the same gradient parameters")

    if props is not None:
        for parameter, grad_a in a.gradients():
            grad_b = b.gradient(parameter)
            for prop in props:
                err_msg_len = (
                    f"gradient '{parameter}' {prop} of the two `TensorBlock` "
                    "have different lengths"
                )

                err_msg_1 = (
                    f"gradient '{parameter}' {prop} are not the same or not in the "
                    "same order"
                )

                err_msg_names = (
                    f"gradient '{parameter}' {prop} names are not the same "
                    "or not in the same order"
                )

                if prop == "samples":
                    if not len(grad_a.samples) == len(grad_b.samples):
                        raise ValueError(err_msg + err_msg_len)
                    if not grad_a.samples.names == grad_b.samples.names:
                        raise ValueError(err_msg + err_msg_names)
                    if not np.all(grad_a.samples == grad_b.samples):
                        raise ValueError(err_msg + err_msg_1)
                elif prop == "properties":
                    if not len(grad_a.properties) == len(grad_b.properties):
                        raise ValueError(err_msg + err_msg_len)
                    if not grad_a.properties.names == grad_b.properties.names:
                        raise ValueError(err_msg + err_msg_names)
                    if not np.all(grad_a.properties == grad_b.properties):
                        raise ValueError(err_msg + err_msg_1)
                elif prop == "components":
                    if len(grad_a.components) != len(grad_b.components):
                        raise ValueError(err_msg + err_msg_len)

                    for c1, c2 in zip(grad_a.components, grad_b.components):
                        if not (c1.names == c2.names):
                            raise ValueError(err_msg + err_msg_names)

                        if not np.all(c1 == c2):
                            raise ValueError(err_msg + err_msg_1)
                elif prop != "parameters":
                    # parameters are already checked at the beginning but i want
                    # to give the opportunity to the 'user' to use it, without
                    # problems
                    raise ValueError(
                        f"{prop} is not a valid property to check, "
                        "choose from ['samples', 'properties', 'components']"
                    )


def _check_gradient_presence(block: TensorBlock, parameters: List, fname: str):
    for p in parameters:
        if p not in block.gradients_list():
            raise ValueError(
                f"requested gradient '{p}' in {fname} is not defined in this tensor"
            )


def _labels_equal(a: Labels, b: Labels, exact_order: bool):
    """
    For 2 :py:class:`Labels` objects ``a`` and ``b``, returns true if they are
    exactly equivalent in names, values, and elemental positions. Assumes that
    the Labels are already searchable, i.e. they belong to a parent TensorBlock
    or TensorMap.
    """
    # They can only be equivalent if the same length
    if len(a) != len(b):
        return False
    # Check the names
    if not np.all(a.names == b.names):
        return False
    if exact_order:
        return np.all(np.array(a == b))
    return np.all([a_i in b for a_i in a])
