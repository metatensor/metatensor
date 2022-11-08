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
    This function has the equivalent behaviour of  ``np.dot(array1,array2.T)``.
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
            return np.dot(array1, np.transpose(array2))
    elif isinstance(array1, TorchTensor) and isinstance(array2, TorchTensor):
        shape1 = array1.size()
        shape2 = array2.size()
        if len(shape1) == 1 and len(shape2) == 1:
            return torch.dot(array1, array2.T)
        else:
            return array1 @ array2.T
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def solve(array1, array2):
    """Computes the solution of a square system of linear equations
    with a unique solution.
    This function has the equivalent
    behaviour of  ``numpy.linalg.solve(array1,array2)``.
    """
    if isinstance(array1, np.ndarray) and isinstance(array2, np.ndarray):
        return np.linalg.solve(array1, array2)
    elif isinstance(array1, TorchTensor) and isinstance(array2, TorchTensor):
        result = torch.linalg.solve(array1, array2)
        return result
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def lstsq(array1, array2, rcond, driver=None):
    """
    Computes a solution to the least squares problem
    of a system of linear equations.
    Computes the vector x that approximately
    solves the equation ``array1 @ x = array2``.
    This function has the equivalent
    behaviour of ``numpy.linalg.lstsq(array1,array2)``.

    rcond: Cut-off ratio for small singular values of a.
    WARNING: the default rcond=None for numpy and torch is different
    numpy -> rcond is the machine precision times max(M, N).
             with M, N being the dimensions of array1
    torch -> rcond is the machine precision,
             to have this behaviour in numpy use
             rcond=-1

    driver: Used only in torch (ignored if numpy id used).
            It chooses the LAPACK/MAGMA function that will be used.
            Possible values: for CPU ‘gels’, ‘gelsy’, ‘gelsd, ‘gelss’.
                             for GPU  the only valid driver is ‘gels’,
                             which assumes that A is full-rank
            see https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html
            for a full description
            If None, ‘gelsy’ is used for CPU inputs
            and ‘gels’ for CUDA inputs. Default: None
    """
    if isinstance(array1, np.ndarray) and isinstance(array2, np.ndarray):
        return np.linalg.lstsq(array1, array2, rcond=rcond)[0]
    elif isinstance(array1, TorchTensor) and isinstance(array2, TorchTensor):
        result = torch.linalg.lstsq(array1, array2, rcond=rcond, driver=driver)[0]
        return result
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def vstack(arrays):
    """Stack vertically a group of arrays.
    This function has the equivalent
    behaviour of ``numpy.vstack(arrays)``.

    Args:
        arrays : sequence of arrays

    Returns:
        array : vertical-stacked array
    """
    if isinstance(arrays[0], np.ndarray):
        return np.vstack(arrays)
    elif isinstance(arrays[0], TorchTensor):
        return torch.vstack(arrays)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)
