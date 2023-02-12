from typing import List

import numpy as np

from ..block import TensorBlock
from ..labels import Labels
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
        err_msg = f"Inputs to {fname} should have the same {prop}."
        if prop == "samples":
            if not a.samples.names == b.samples.names:
                raise ValueError(err_msg)
            if not np.all(a.samples == b.samples):
                raise ValueError(err_msg)
        elif prop == "properties":
            if not a.properties.names == b.properties.names:
                raise ValueError(err_msg)
            if not np.all(a.properties == b.properties):
                raise ValueError(err_msg)
        elif prop == "components":
            if len(a.components) != len(b.components):
                raise ValueError(err_msg)

            for ca in a.components:
                flag = False
                for cb in b.components:
                    if ca.names == cb.names:
                        flag = True
                        if not np.all(ca == cb):
                            raise ValueError(err_msg)
                # if flag is False ca.names == cb.names has never been satisfied
                if not flag:
                    raise ValueError(err_msg)

        elif prop == "gradients":
            grads_a = a.gradients_list()
            grads_b = b.gradients_list()

            if len(grads_a) != len(grads_b) or (
                not np.all([parameter in grads_b for parameter in grads_a])
            ):
                raise ValueError(err_msg)
        else:
            raise ValueError(
                f"{prop} is not a valid property to check. "
                "Choose from 'samples', 'properties', 'components' "
                "or 'gradients'."
            )


def _check_same_gradients_components(a: TensorBlock, b: TensorBlock, fname: str):
    for parameter, grad_a in a.gradients():
        grad_b = b.gradient(parameter)
        grad_comps_a = np.array(grad_a.components, dtype=object)
        grad_comps_b = np.array(grad_b.components, dtype=object)

        if not np.all(grad_comps_a == grad_comps_b):
            raise ValueError(
                f"input to {fname} should habe the same gradient components"
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
    else:
        return np.all([a_i in b for a_i in a])
