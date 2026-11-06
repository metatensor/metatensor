"""
Small set of functions working on both numpy arrays and torch tensors, used to
manipulate Labels values without moving them to another device.
"""

import numpy as np

from ._data._array import is_numpy_array, is_torch_array


try:
    import torch
except ImportError:
    pass


UNKNOWN_ARRAY_TYPE = (
    "unknown array type, only numpy arrays and torch tensors are supported"
)


def column_like(values, like):
    """
    Convert the 1D ``values`` to 32-bit integers on the same device as ``like``. Both
    arrays must use the same array type.
    """
    if is_numpy_array(like):
        return values.astype(np.int32, copy=False)
    elif is_torch_array(like):
        return values.to(dtype=torch.int32, device=like.device)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def insert_column(array, index: int, column):
    """
    Insert ``column`` in the 2D ``array`` before the column at ``index``. The column
    must use the same array type, dtype and device as ``array``.

    This is equivalent to ``numpy.insert(array, index, column, axis=1)``.
    """
    if is_numpy_array(array):
        return np.insert(array, index, column, axis=1)
    elif is_torch_array(array):
        return torch.cat(
            [array[:, :index], column.reshape(-1, 1), array[:, index:]], dim=1
        )
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def remove_column(array, index: int):
    """
    Remove the column at ``index`` from the 2D ``array``.

    This is equivalent to ``numpy.delete(array, index, axis=1)``.
    """
    if is_numpy_array(array):
        return np.delete(array, index, axis=1)
    elif is_torch_array(array):
        return torch.cat([array[:, :index], array[:, index + 1 :]], dim=1)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)
