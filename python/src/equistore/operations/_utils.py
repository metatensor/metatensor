from typing import List

import numpy as np

from ..block import TensorBlock
from ..tensor import TensorMap


def _check_same_keys(a: TensorMap, b: TensorMap, fname: str):
    keys_a = a.keys
    keys_b = b.keys

    if len(keys_a) != len(keys_b) or (not np.all([key in keys_b for key in keys_a])):
        raise ValueError(f"inputs to {fname} should have the same keys")


def _check_blocks(a: TensorBlock, b: TensorBlock, props: List[str], fname: str):
    """Check if a proprty is the same between two :py:class:`TensorBlock`s.

    :param a: first :py:class:`TensorBlock` for check
    :param b: second :py:class:`TensorBlock` for check
    :param props: A list of strings containing the property to check.
                 Allowed values are ``'properties'`` or ``'samples'``,
                 ``'components'`` and ``'gradients'``.
    """
    for prop in props:
        err_msg = f"inputs to {fname} should have the same {prop}"
        if prop == "samples":
            if not np.all(a.samples == b.samples):
                raise ValueError(err_msg)
        elif prop == "properties":
            if not np.all(a.properties == b.properties):
                raise ValueError(err_msg)
        elif prop == "components":
            if not np.all(a.components == b.components):
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
