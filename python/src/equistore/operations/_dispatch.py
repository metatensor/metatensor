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


def _check_all_same_type(arrays, expected_type):
    for array in arrays:
        if not isinstance(array, expected_type):
            raise TypeError(
                f"expected argument to be a {expected_type}, but got {type(array)}"
            )


def allclose(a, b, rtol, atol, equal_nan=False):
    """Compare two arrays using ``allclose``

    This function has the same behavior as
    ``np.allclose(array1, array2, rtol, atol, equal_nan)``.
    """
    if isinstance(a, np.ndarray):
        _check_all_same_type([b], np.ndarray)
        return np.allclose(a=a, b=b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    elif isinstance(a, TorchTensor):
        _check_all_same_type([b], TorchTensor)
        return torch.allclose(
            input=a, other=b, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def bincount(input, weights=None, minlength=0):
    """Count number of occurrences of each value in array of non-negative ints.
    Equivalent of ``numpy.bitcount(input, weights, minlength)``

    Args:
        input (array_like): Input array.
        weights (array_like, optional): Weights, array of the same shape as input.
                                        Defaults to None.
        minlength (int, optional): A minimum number of bins for the output array.
                                        Defaults to 0.
    """
    if isinstance(input, np.ndarray):
        return np.bincount(input, weights=weights, minlength=minlength)
    elif isinstance(input, TorchTensor):
        return torch.bincount(input, weights=weights, minlength=minlength)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


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


def dot(A, B):
    """Compute dot product of two arrays.

    This function has the same behavior as  ``np.dot(A, B.T)``, and assumes the
    second array is 2-dimensional.
    """
    if isinstance(A, np.ndarray):
        _check_all_same_type([B], np.ndarray)
        shape1 = A.shape
        assert len(B.shape) == 2
        # Using matmul/@ is the recommended way in numpy docs for 2-dimensional
        # matrices
        if len(shape1) == 2:
            return A @ B.T
        else:
            return np.dot(A, B.T)
    elif isinstance(A, TorchTensor):
        _check_all_same_type([B], TorchTensor)
        assert len(B.shape) == 2
        return A @ B.T
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def solve(X, Y):
    """
    Computes the solution of a square system of linear equations with a unique
    solution.

    This function has the same behavior as ``numpy.linalg.solve(X, Y)``.
    """
    if isinstance(X, np.ndarray):
        _check_all_same_type([Y], np.ndarray)
        return np.linalg.solve(X, Y)
    elif isinstance(X, TorchTensor):
        _check_all_same_type([Y], TorchTensor)
        result = torch.linalg.solve(X, Y)
        return result
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def lstsq(X, Y, rcond, driver=None):
    """
    Computes a solution to the least squares problem of a system of linear
    equations.

    Computes the vector x that approximately solves the equation ``array1 @ x =
    array2``. This function has the same behavior as ``numpy.linalg.lstsq(X,
    Y)``.

    :param rcond: Cut-off ratio for small singular values of a.
        WARNING: the default rcond=None for numpy and torch is different
        numpy -> rcond is the machine precision times max(M, N).
                with M, N being the dimensions of array1
        torch -> rcond is the machine precision,
                to have this behavior in numpy use
                rcond=-1

    :param driver: Used only in torch (ignored if numpy is used).
            Chooses the LAPACK/MAGMA function that will be used.
            Possible values: for CPU 'gels', 'gelsy', 'gelsd', 'gelss'.
                             for GPU  the only valid driver is 'gels',
                             which assumes that A is full-rank
            see https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html
            for a full description
            If None, 'gelsy' is used for CPU inputs
            and 'gels' for CUDA inputs. Default: None
    """
    if isinstance(X, np.ndarray):
        _check_all_same_type([Y], np.ndarray)
        return np.linalg.lstsq(X, Y, rcond=rcond)[0]
    elif isinstance(X, TorchTensor):
        _check_all_same_type([Y], TorchTensor)
        result = torch.linalg.lstsq(X, Y, rcond=rcond, driver=driver)[0]
        return result
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def hstack(arrays):
    """Stack horizontally a group of arrays.

    This function has the same behavior as ``numpy.hstack(arrays)``.
    """
    if isinstance(arrays[0], np.ndarray):
        _check_all_same_type(arrays, np.ndarray)
        return np.hstack(arrays)
    elif isinstance(arrays[0], TorchTensor):
        _check_all_same_type(arrays, TorchTensor)
        return torch.hstack(arrays)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def vstack(arrays):
    """Stack vertically a group of arrays.

    This function has the same behavior as ``numpy.vstack(arrays)``.
    """
    if isinstance(arrays[0], np.ndarray):
        _check_all_same_type(arrays, np.ndarray)
        return np.vstack(arrays)
    elif isinstance(arrays[0], TorchTensor):
        _check_all_same_type(arrays, TorchTensor)
        return torch.vstack(arrays)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def index_add(output_array, input_array, index):
    """Accumulates in `output_array`
    the elements of `array`
    by adding to the indices in the order given in index.

    it is equivalent of torch's:

    output_array.index_add_(0, torch.tensor(index),input_array)

    """
    if len(index.shape) != 1:
        raise ValueError("index should be 1D array")
    if isinstance(input_array, np.ndarray):
        _check_all_same_type([output_array], np.ndarray)
        np.add.at(output_array, index, input_array)
    elif isinstance(input_array, TorchTensor):
        _check_all_same_type([output_array], TorchTensor)
        output_array.index_add_(0, torch.tensor(index), input_array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def zeros(shape, array_type="numpy"):
    """Create a `zeros_like` with the given input shape.

    `array_type` can be `Numpy` or `torch`.
    """
    if array_type == "numpy":
        return np.zeros(shape)
    elif array_type == "torch":
        return torch.zeros(shape)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def zeros_like(array, shape=None, requires_grad=False):
    """Create an zeros_like with the same size of array.
    if shape is not None it overrides the shape of the result.

    It is equivalent of np.zeros_like(array, shape=shape).
    requires_grad is used only in torch"""
    if isinstance(array, np.ndarray):
        return np.zeros_like(array, shape=shape)
    elif isinstance(array, TorchTensor):
        if shape is None:
            shape = array.size()

        return torch.zeros(
            shape,
            dtype=array.dtype,
            layout=array.layout,
            device=array.device,
            requires_grad=requires_grad,
        )
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def ones_like(array, shape=None, requires_grad=False):
    """Create an ones_like with the same size of array.

    :param shape: If not ``None`` override the shape with the given one
    :param requires_grad: used only in torch
    """
    if isinstance(array, np.ndarray):
        return np.ones_like(array, shape=shape)
    elif isinstance(array, TorchTensor):
        if shape is None:
            shape = array.size()

        return torch.ones_like(
            shape,
            dtype=array.dtype,
            layout=array.layout,
            device=array.device,
            requires_grad=requires_grad,
        )
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def empty_like(array, shape=None, requires_grad=False):
    """Create an empty_like with the same size of array.
    if shape is not None it overrides the shape of the result.

    It is equivalent of np.empty_like(array, shape=shape).
    requires_grad is used only in torch"""
    if isinstance(array, np.ndarray):
        return np.empty_like(array, shape=shape)
    elif isinstance(array, TorchTensor):
        if shape is None:
            shape = array.size()

        return torch.empty_like(
            shape,
            dtype=array.dtype,
            layout=array.layout,
            device=array.device,
            requires_grad=requires_grad,
        )
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)
