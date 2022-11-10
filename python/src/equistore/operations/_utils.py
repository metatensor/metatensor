import numpy as np

from ..tensor import TensorMap
from ..block import TensorBlock


def _check_same_keys(a: TensorMap, b: TensorMap, fname: str):
    keys_a = a.keys
    keys_b = b.keys

    if len(keys_a) != len(keys_b) or (not np.all([key in keys_b for key in keys_a])):
        raise ValueError(f"inputs to {fname} should have the same keys")


def _check_same_gradients(a: TensorBlock, b: TensorBlock, fname: str):
    grad_a = a.gradients_list()
    grad_b = b.gradients_list()

    if len(grad_a) != len(grad_b) or (
        not np.all([parameter in grad_b for parameter in grad_a])
    ):
        raise ValueError(f"inputs to {fname} should have the same gradients")
