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


def dot(array1, array2):
    """Compute dot product of two arrays.
    This function has the equivalent bheaviour of  ``np.dot(array1,array2)``.
    For numpy array, it check if the arrays are 2D, in which case it uses the matmul,
    which is preferred according to the doc.
    For torch it uses ``torch.dot``
    if the arrays are 1D, ``torch.matmul`` is used otherwise.
    """
    if isinstance(array1, np.ndarray) and isinstance(array2, np.ndarray):
        shape1 = array1.shape
        shape2 = array2.shape
        if len(shape1) == 2 and len(shape2) == 2:
            return array1 @ array2.T
        else:
            return np.dot(array1, array2.T)
    elif isinstance(array1, TorchTensor) and isinstance(array2, TorchTensor):
        shape1 = array1.size()
        shape2 = array2.size()
        if len(shape1) == 1 and len(shape2) == 1:
            return torch.dot(array1, array2)
        else:
            return torch.matmul(array1, torch.Traspose(array2))
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)
