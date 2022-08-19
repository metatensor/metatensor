import numpy as np

from ..data import HAS_TORCH

if HAS_TORCH:
    import torch
    from torch import Tensor as TorchTensor
else:

    class TorchTensor:
        pass


UNKNOWN_ARRAY_TYPE = (
    "unknown array type, only numpy arrays and torch tensors are supported"
)


def norm(array, axis=None):
    """Compute the 2-norm (Frobenius norm for matrices) of the input array.

    This calls the equivalent of ``np.linalg.norm(array, axis=axis)``, see this
    function for more documentation.
    """
    if isinstance(array, np.ndarray):
        return np.linalg.norm(array, axis=axis)
    elif isinstance(array, TorchTensor):
        return torch.linalg.norm(array, dim=axis)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)
