from typing import List, Union

import numpy as np

from equistore.core import TensorBlock, TensorMap


class NotEqualError(Exception):
    """Exception used to indicate that two equistore objects are different"""

    pass


def _check_same_keys(a: TensorMap, b: TensorMap, fname: str) -> bool:
    return not bool(_check_same_keys_impl(a, b, fname))


def _check_same_keys_raise(a: TensorMap, b: TensorMap, fname: str) -> None:
    message = _check_same_keys_impl(a, b, fname)
    if message != "":
        raise NotEqualError(message)


def _check_same_keys_impl(a: TensorMap, b: TensorMap, fname: str) -> str:
    """Check if metadata between two TensorMaps is consistent for an operation.

    The functions verifies that

    1. The key names are the same.
    2. The number of blocks in the same
    3. The block key indices are the same.

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


def _check_blocks(a: TensorBlock, b: TensorBlock, props: List[str], fname: str) -> str:
    return not bool(_check_blocks_impl(a, b, props, fname))


def _check_blocks_raise(
    a: TensorBlock, b: TensorBlock, props: List[str], fname: str
) -> None:
    message = _check_blocks_impl(a, b, props, fname)
    if message != "":
        raise NotEqualError(message)


def _check_blocks_impl(
    a: TensorBlock, b: TensorBlock, props: List[str], fname: str
) -> str:
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
                return err_msg + err_msg_len
            if not a.samples.names == b.samples.names:
                return err_msg + err_msg_names
            if not np.all(a.samples == b.samples):
                return err_msg + err_msg_1

        elif prop == "properties":
            if not len(a.properties) == len(b.properties):
                return err_msg + err_msg_len
            if not a.properties.names == b.properties.names:
                return err_msg + err_msg_names
            if not np.all(a.properties == b.properties):
                return err_msg + err_msg_1

        elif prop == "components":
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
                f"{prop} is not a valid property to check, "
                "choose from ['samples', 'properties', 'components']"
            )
    return ""


def _check_same_gradients(
    a: TensorBlock, b: TensorBlock, props: List[str], fname: str
) -> bool:
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
    return not bool(_check_same_gradients_impl(a, b, props, fname))


def _check_same_gradients_raise(
    a: TensorBlock, b: TensorBlock, props: List[str], fname: str
) -> None:
    """
    Check if metadata between two gradients's TensorBlocks is consistent for an
    operation.

    The message associated with the exception will contain more information on where the
    gradients of the two :py:class:`TensorBlock` differ. See
    :py:func:`_check_same_gradients` for more information on when gradients of
    :py:class:`TensorBlock` are considered as equal.

    :raises: :py:class:`equistore.NotEqualError` if the gradients of the blocks are
        different

    :param a: first :py:class:`TensorBlock` for check
    :param b: second :py:class:`TensorBlock` for check
    :param props: A list of strings containing the property to check. Allowed values are
                 ``'samples'`` or ``'properties'``, ``'components'``. To check only if
                 the ``'parameters'`` are consistent use ``props=None``, using
                 ``'parameters'`` is allowed
                  even though deprecated.

    :raises: :py:class:`NotImplementedError` if one of :py:class:`TensorBlock` inputs
             has gradients of gradients
    :raises: :py:class:`ValueError` if an invalid prop name in :param props: is given.
             See :param props: description for valid prop names
    """
    message = _check_same_gradients_impl(a, b, props, fname)
    if message != "":
        raise NotEqualError(message)


def _check_same_gradients_impl(
    a: TensorBlock, b: TensorBlock, props: List[str], fname: str
) -> Union[str, None]:
    err_msg = f"inputs to {fname} should have the same gradients:\n"
    gradients_list_a = a.gradients_list()
    gradients_list_b = b.gradients_list()

    if len(gradients_list_a) != len(gradients_list_b) or (
        not np.all([parameter in gradients_list_b for parameter in gradients_list_a])
    ):
        return f"inputs to {fname} should have the same gradient parameters"

    for parameter, grad_a in a.gradients():
        grad_b = b.gradient(parameter)

        _check_same_gradients(grad_a, grad_b, props=props, fname=fname)

        if props is not None:
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
                        return err_msg + err_msg_len
                    if not grad_a.samples.names == grad_b.samples.names:
                        return err_msg + err_msg_names
                    if not np.all(grad_a.samples == grad_b.samples):
                        return err_msg + err_msg_1

                elif prop == "properties":
                    if not len(grad_a.properties) == len(grad_b.properties):
                        return err_msg + err_msg_len
                    if not grad_a.properties.names == grad_b.properties.names:
                        return err_msg + err_msg_names
                    if not np.all(grad_a.properties == grad_b.properties):
                        return err_msg + err_msg_1
                elif prop == "components":
                    if len(grad_a.components) != len(grad_b.components):
                        return err_msg + err_msg_len

                    for c1, c2 in zip(grad_a.components, grad_b.components):
                        if not (c1.names == c2.names):
                            return err_msg + err_msg_names

                        if not np.all(c1 == c2):
                            return err_msg + err_msg_1

                elif prop != "parameters":
                    # parameters are already checked at the beginning but i want
                    # to give the opportunity to the 'user' to use it, without
                    # problems
                    raise ValueError(
                        f"{prop} is not a valid property to check, "
                        "choose from ['samples', 'properties', 'components']"
                    )
    return ""


def _check_gradient_presence(block: TensorBlock, parameters: List[str], fname: str):
    for parameter in parameters:
        if parameter not in block.gradients_list():
            raise ValueError(
                f"requested gradient '{parameter}' in {fname} is not defined "
                "in this tensor"
            )
