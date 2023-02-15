from typing import List

import numpy as np

from ..block import TensorBlock
from ..tensor import TensorMap


def _check_maps(a: TensorMap, b: TensorMap, fname: str):
    """Check if metadata between two TensorMaps is consistent for an operation.

    The functions verifies that

    1. The key names are the same.
    2. The number of blocks in the same
    3. The block key indices are the same.

    :param a: first :py:class:`TensorMap` for check
    :param b: second :py:class:`TensorMap` for check
    """

    if a.keys.names != b.keys.names:
        raise ValueError(
            f"Inputs to {fname} should have the same keys. "
            f"Got {a.keys.names} and {b.keys.names}."
        )

    if len(a.blocks()) != len(b.blocks()):
        raise ValueError(
            f"Inputs to {fname} should have the same number of blocks. "
            f"Got {len(a.blocks())} and {len(b.blocks())}."
        )

    if not np.all([key in a.keys for key in b.keys]):
        raise ValueError(f"Inputs to {fname} should have the same key indices.")


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
        err_msg = f"Inputs to '{fname}' should have the same {prop}:\n"
        err_msg_len = f"{prop} of the two `TensorBlock` have not the same lenght."
        err_msg_1 = f"{prop} are not the same or not in the same order."
        err_msg_names = f"{prop} names are not the same or not in the same order."
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

                if not np.all(c1 == c2):
                    raise ValueError(err_msg + err_msg_1)

        else:
            raise ValueError(
                f"{prop} is not a valid property to check. "
                "Choose from 'samples', 'properties', 'components'."
            )


def _check_same_gradients(a: TensorBlock, b: TensorBlock, props: List[str], fname: str):
    """Check if metadata between two gradients's TensorBlocks is consistent
     for an operation.

    The functions verifies that the metadata of the given props is the same
    (length, names, values) and in the same order.

    If props is None it only checks if the ``'parameters'`` are consistent.

    :param a: first :py:class:`TensorBlock` for check
    :param b: second :py:class:`TensorBlock` for check
    :param props: A list of strings containing the property to check.
                 Allowed values are ``'samples'`` or ``'properties'``,
                 ``'components'``. To check only if the ``'parameters'`` are consistent
                 use ``props=None``, using ``'parameters'`` is allowed
                  even though deprecated.
    """
    err_msg = f"Inputs to {fname} should have the same gradient:\n"
    grads_a = a.gradients_list()
    grads_b = b.gradients_list()

    if len(grads_a) != len(grads_b) or (
        not np.all([parameter in grads_b for parameter in grads_a])
    ):
        raise ValueError(f"Inputs to {fname} should have the same gradient parameters.")

    if props is not None:
        for parameter, grad_a in a.gradients():
            grad_b = b.gradient(parameter)
            for prop in props:
                err_msg_len = (
                    f'gradient ("{parameter}") {prop} of the two `TensorBlock` '
                    "have not the same lenght."
                )

                err_msg_1 = (
                    f'gradient ("{parameter}") {prop} are not the same or not in the '
                    "same order."
                )

                err_msg_names = (
                    f'gradient ("{parameter}") {prop} names are not the same '
                    "or not in the same order."
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
                    # parameters are already checked at the begining but i want to give
                    # the opportunity to the 'user' to use it, without problems
                    raise ValueError(
                        f"{prop} is not a valid property of the gradients to check. "
                        "Choose from 'parameters', 'samples', "
                        "'properties', 'components'."
                    )


def _check_parameters_in_gradient_block(
    block: TensorBlock, parameters: List, fname: str
):
    for p in parameters:
        if p not in block.gradients_list():
            raise ValueError(
                f"The requested parameter '{p}' in {fname} is not a valid parameter"
                "for the TensorBlock"
            )
