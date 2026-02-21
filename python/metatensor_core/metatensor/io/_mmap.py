import ctypes

import numpy as np

from .._c_lib import _get_library
from ..utils import catch_exceptions


class MmapOwner:
    """Internal helper to keep the memory mapping alive."""

    def __init__(self, mmap_ptr):
        self._ptr = mmap_ptr

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _get_library().mts_mmap_free(self._ptr)
            self._ptr = None


class _MmapNdarray(np.ndarray):
    """Thin ndarray subclass that holds a reference to an MmapOwner."""

    def __new__(cls, array, owner):
        obj = np.asarray(array).view(cls)
        obj._mmap_owner = owner
        return obj

    def __array_finalize__(self, obj):
        self._mmap_owner = getattr(obj, "_mmap_owner", None)


@catch_exceptions
def _create_mmap_array(
    shape_ptr, shape_count, dtype, data_ptr, data_len, mmap_ptr, array_ptr
):
    from ..data.array import create_mts_array
    from ..data.extract import _DLPACK_TO_NUMPY, _ptr_to_ndarray

    shape = []
    for i in range(shape_count):
        shape.append(shape_ptr[i])

    owner = MmapOwner(mmap_ptr)

    np_dtype = _DLPACK_TO_NUMPY.get((dtype.code, dtype.bits))
    if np_dtype is None:
        raise ValueError(
            f"unsupported DLPack dtype: code={dtype.code}, bits={dtype.bits}"
        )

    c_type = np.ctypeslib.as_ctypes_type(np.dtype(np_dtype))
    array = _ptr_to_ndarray(
        ctypes.cast(data_ptr, ctypes.POINTER(c_type)),
        shape,
        np_dtype,
    )

    # Check alignment
    if (data_ptr % array.dtype.alignment) != 0:
        # Not aligned, must copy (owner still freed when MmapOwner is GC'd)
        array = array.copy()
    else:
        # Wrap in _MmapNdarray to keep the MmapOwner alive via reference
        array = _MmapNdarray(array, owner)

    array_ptr[0] = create_mts_array(array)
