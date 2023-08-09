from typing import List, Sequence

import numpy as np

from equistore.core import TensorBlock, TensorMap


class NotEqualError(Exception):
    """Exception used to indicate that two equistore objects are different"""

    pass


def _check_same_keys(a: TensorMap, b: TensorMap, fname: str) -> bool:
    """
    Returns true if the keys of 2 TensorMaps are the same, without specification of the
    order, and false otherwise.
    """
    return not bool(_check_same_keys_impl(a, b, fname))


def _check_same_keys_raise(a: TensorMap, b: TensorMap, fname: str) -> None:
    """
    If the keys of 2 TensorMaps are not the same, raises a NotEqualError, othwerwise
    returns None.
    """
    message = _check_same_keys_impl(a, b, fname)
    if message != "":
        raise NotEqualError(message)


def _check_same_keys_impl(a: TensorMap, b: TensorMap, fname: str) -> str:
    """
    Checks if the keys of 2 TensorMaps are the same, without specification of the order.
    Returns an empty str if they are the same, otherwise returns a str message of a
    meaningful error.

    The functions verifies that the key names and length are the same, and that the key
    values are the same without specification of exact order.

    :param a: first :py:class:`TensorMap` for check
    :param b: second :py:class:`TensorMap` for check
    :param fname: name of the function where the check is performed. The input will be
        used to generate a meaningful error message.
    """

    keys_a = a.keys
    keys_b = b.keys

    if keys_a.names != keys_b.names:
        return (
            f"inputs to {fname} should have the same keys names, "
            f"got '{keys_a.names}' and '{keys_b.names}'"
        )

    if len(keys_a) != len(keys_b):
        return (
            f"inputs to {fname} should have the same number of blocks, "
            f"got {len(keys_a)} and {len(keys_b)}"
        )

    if not np.all([key in keys_a for key in keys_b]):
        return f"inputs to {fname} should have the same keys"

    return ""


def _check_blocks(
    a: TensorBlock,
    b: TensorBlock,
    fname: str,
    check: Sequence[str] = ("samples", "components", "properties"),
) -> bool:
    """
    Checks if the metadata of 2 TensorBlocks are the same. If not, returns false.
    Otherwise returns true.
    """
    return not bool(_check_blocks_impl(a, b, fname, check))


def _check_blocks_raise(
    a: TensorBlock,
    b: TensorBlock,
    fname: str,
    check: Sequence[str] = ("samples", "components", "properties"),
) -> None:
    """
    Checks if the metadata of two TensorBlocks are the same. If not, raises a
    NotEqualError. If invalid `check` are given, this function raises a `ValueError`.
    Otherwise returns it None.

    The message associated with the exception will contain more information on where the
    two :py:class:`TensorBlock` differ. See :py:func:`_check_blocks_impl` for more
    information on when two :py:class:`TensorBlock` are considered as equal.

    :param a: first :py:class:`TensorBlock` for check
    :param b: second :py:class:`TensorBlock` for check
    :param fname: name of the function where the check is performed. The input will be
        used to generate a meaningful error message.
    :param check: A sequence of strings containing the metadata to check. Allowed values
        are ``'properties'`` or ``'samples'``, ``'components'``.

    :raises: :py:class:`equistore.NotEqualError` if the metadata of the blocks are
        different
    :raises: :py:class:`ValueError` if an invalid prop name in :param check: is
        given. See :param check: description for valid prop names
    """
    message = _check_blocks_impl(a, b, fname, check)
    if message != "":
        raise NotEqualError(message)


def _check_blocks_impl(
    a: TensorBlock,
    b: TensorBlock,
    fname: str,
    check: Sequence[str] = ("samples", "components", "properties"),
) -> str:
    """
    Check if metadata between two TensorBlocks is consistent for an operation.

    The functions verifies that that the metadata of the given check is the same, in
    terms of length, dimension names, and order of the values.

    If they are not the same, an error message as a str is returned. Otherwise, an empty
    str is returned.

    :param a: first :py:class:`TensorBlock` for check
    :param b: second :py:class:`TensorBlock` for check
    :param fname: name of the function where the check is performed. The input will be
        used to generate a meaningful error message.
    :param check: A sequence of strings containing the metadata to check. Allowed values
        are ``'properties'`` or ``'samples'``, ``'components'``.
    """
    for metadata in check:
        err_msg = f"inputs to '{fname}' should have the same {metadata}:\n"
        err_msg_len = f"{metadata} of the two `TensorBlock` have different lengths"
        err_msg_1 = f"{metadata} are not the same or not in the same order"
        err_msg_names = f"{metadata} names are not the same or not in the same order"

        if metadata == "samples":
            if not len(a.samples) == len(b.samples):
                return err_msg + err_msg_len
            if not a.samples.names == b.samples.names:
                return err_msg + err_msg_names
            if not np.all(a.samples == b.samples):
                return err_msg + err_msg_1

        elif metadata == "properties":
            if not len(a.properties) == len(b.properties):
                return err_msg + err_msg_len
            if not a.properties.names == b.properties.names:
                return err_msg + err_msg_names
            if not np.all(a.properties == b.properties):
                return err_msg + err_msg_1

        elif metadata == "components":
            if len(a.components) != len(b.components):
                return err_msg + err_msg_len

            for c1, c2 in zip(a.components, b.components):
                if not (c1.names == c2.names):
                    return err_msg + err_msg_names

                if not (len(c1) == len(c2)):
                    return err_msg + err_msg_1

                if not np.all(c1 == c2):
                    return err_msg + err_msg_1
        else:
            raise ValueError(
                f"{metadata} is not a valid property to check, "
                "choose from 'samples', 'properties' and 'components'"
            )
    return ""


def _check_same_gradients(
    a: TensorBlock,
    b: TensorBlock,
    fname: str,
    check: Sequence[str] = ("samples", "components", "properties"),
) -> bool:
    """
    Check if metadata between the gradients of 2 TensorBlocks is consistent for an
    operation. If they are the same, true is returned, otherwise false.

    The functions verifies that that the metadata of the given check is the same, in
    terms of length, dimension names, and order of the values.

    If check is None it only checks if the ``'parameters'`` are consistent.

    :param a: first :py:class:`TensorBlock` for check
    :param b: second :py:class:`TensorBlock` for check
    :param fname: name of the function where the check is performed. The input will be
        used to generate a meaningful error message.
    :param check: A sequence of strings containing the metadata to check. Allowed values
        are ``'properties'`` or ``'samples'``, ``'components'``. To check only if the
        ``'parameters'`` are consistent pass an empty tuple ``check=()``.
    """
    return not bool(_check_same_gradients_impl(a, b, fname, check))


def _check_same_gradients_raise(
    a: TensorBlock,
    b: TensorBlock,
    fname: str,
    check: Sequence[str] = ("samples", "components", "properties"),
) -> None:
    """
    Check if two TensorBlocks gradients have identical metadata.

    The message associated with the exception will contain more information on where the
    gradients of the two :py:class:`TensorBlock` differ. See
    :py:func:`_check_same_gradients_impl` for more information on when gradients of
    :py:class:`TensorBlock` are considered as equal.

    :param a: first :py:class:`TensorBlock` for check
    :param b: second :py:class:`TensorBlock` for check
    :param fname: name of the function where the check is performed. The input will be
        used to generate a meaningful error message.
    :param check: A sequence of strings containing the metadata to check. Allowed values
        are ``'properties'`` or ``'samples'``, ``'components'``. To check only if the
        ``'parameters'`` are consistent pass an empty tuple ``check=()``.
    :raises: :py:class:`equistore.NotEqualError` if the gradients of the blocks are
        different
    :raises: :py:class:`ValueError` if an invalid prop name in :param check: is
        given. See :param check: description for valid prop names
    """
    message = _check_same_gradients_impl(a, b, fname, check)
    if message != "":
        raise NotEqualError(message)


def _check_same_gradients_impl(
    a: TensorBlock,
    b: TensorBlock,
    fname: str,
    check: Sequence[str] = ("samples", "components", "properties"),
) -> str:
    """
    Check if metadata between the gradients of two TensorBlocks is consistent for an
    operation.

    The functions verifies that that the 2 TensorBlocks have the same gradient
    parameters, then checks the metadata of the given ``check`` is the same, in terms of
    length, dimension names, and order of the values.

    If they are not the same, an error message as a str is returned. Otherwise, an empty
    str is returned. If the 2 blocks have no gradients, an empty string is returned.

    :param a: first :py:class:`TensorBlock` whose gradients are to be checked
    :param b: second :py:class:`TensorBlock` whose gradients are to be checked
    :param fname: name of the function where the check is performed. The input will be
        used to generate a meaningful error message.
    :param check: A sequence of strings containing the metadata to check. Allowed values
        are ``'properties'`` or ``'samples'``, ``'components'``. To check only if the
        ``'parameters'`` are consistent pass an empty tuple ``check=()``.
    """
    err_msg = f"inputs to {fname} should have the same gradients:\n"
    gradients_list_a = a.gradients_list()
    gradients_list_b = b.gradients_list()

    if len(gradients_list_a) != len(gradients_list_b) or (
        not np.all([parameter in gradients_list_b for parameter in gradients_list_a])
    ):
        return f"inputs to {fname} should have the same gradient parameters"

    for parameter, grad_a in a.gradients():
        grad_b = b.gradient(parameter)

        for metadata in check:
            err_msg_len = (
                f"gradient '{parameter}' {metadata} of the two `TensorBlock` "
                "have different lengths"
            )

            err_msg_1 = (
                f"gradient '{parameter}' {metadata} are not the same or not in the "
                "same order"
            )

            err_msg_names = (
                f"gradient '{parameter}' {metadata} names are not the same "
                "or not in the same order"
            )

            if metadata == "samples":
                if not len(grad_a.samples) == len(grad_b.samples):
                    return err_msg + err_msg_len
                if not grad_a.samples.names == grad_b.samples.names:
                    return err_msg + err_msg_names
                if not np.all(grad_a.samples == grad_b.samples):
                    return err_msg + err_msg_1

            elif metadata == "properties":
                if not len(grad_a.properties) == len(grad_b.properties):
                    return err_msg + err_msg_len
                if not grad_a.properties.names == grad_b.properties.names:
                    return err_msg + err_msg_names
                if not np.all(grad_a.properties == grad_b.properties):
                    return err_msg + err_msg_1
            elif metadata == "components":
                if len(grad_a.components) != len(grad_b.components):
                    return err_msg + err_msg_len

                for c1, c2 in zip(grad_a.components, grad_b.components):
                    if not (c1.names == c2.names):
                        return err_msg + err_msg_names

                    if not np.all(c1 == c2):
                        return err_msg + err_msg_1
            else:
                raise ValueError(
                    f"{metadata} is not a valid property to check, "
                    "choose from 'samples', 'properties' and 'components'"
                )
    return ""


def _check_gradient_presence_raise(
    block: TensorBlock, parameters: List[str], fname: str
) -> None:
    """
    For a single TensorBlock checks if each of the passed ``parameters`` are present as
    parameters of its gradients. If all of them are present, None is returned. Otherwise
    a ValueError is raised.
    """
    for parameter in parameters:
        if parameter not in block.gradients_list():
            raise ValueError(
                f"requested gradient '{parameter}' in {fname} is not defined "
                "in this tensor"
            )
