"""Helper functions to dispatch methods between numpy and torch."""

from typing import List, Union

import numpy as np


try:
    import torch
    from torch import Tensor as TorchTensor
except ImportError:

    class TorchTensor:
        pass


UNKNOWN_ARRAY_TYPE = (
    "unknown array type, only numpy arrays and torch tensors are supported"
)


def int_array_like(int_list: Union[List[int], List[List[int]]], like):
    """
    Converts the input list of int to a numpy array or torch tensor
    based on the type of `like`.
    """
    if isinstance(like, TorchTensor):
        if torch.jit.isinstance(int_list, List[int]):
            return torch.tensor(int_list, dtype=torch.int64, device=like.device)
        else:
            return torch.tensor(int_list, dtype=torch.int64, device=like.device)
    elif isinstance(like, np.ndarray):
        return np.array(int_list).astype(np.int64)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)
