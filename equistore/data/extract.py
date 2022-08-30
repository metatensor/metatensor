import ctypes
from typing import NewType, Union

import numpy as np

from .._c_api import c_uintptr_t, eqs_data_origin_t
from ..utils import _call_with_growing_buffer, _ptr_to_ndarray
from .array import _object_from_ptr, _origin_numpy, _origin_pytorch, _register_origin

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    # This NewType is only used for typechecking and documentation purposes
    Array = NewType("Array", Union[np.ndarray, torch.Tensor])
    """
    An ``Array`` contains the actual data stored in a
    :py:class:`equistore.TensorBlock`.

    This data is manipulated by ``equistore`` in a completely opaque way: this
    library does not know what's inside the arrays appart from a small set of
    constrains:

    - array contains numeric data;
    - they are stored as row-major, n-dimensional arrays with at least 2
      dimensions;
    - it is possible to create new arrays and move data from one array to
      another.

    The actual type of an ``Array`` depends on how the
    :py:class:`equistore.TensorBlock` was created. Currently, numpy ``ndarray``
    and torch ``Tensor`` are supported.
    """

else:
    Array = NewType("Array", np.ndarray)


_ADDITIONAL_ORIGINS = {}


def register_external_data_wrapper(origin, klass):
    """
    Register a non-Python data origin and the corresponding class wrapper.

    The wrapper class constructor must take two arguments (raw ``eqs_array`` and
    python ``parent`` object) and return a subclass of either ``numpy.ndarray``
    or ``torch.Tensor``, which keeps ``parent`` alive.

    :param origin: new origin to register as a string
    :param klass: wrapper class to use for this origin
    """

    if not isinstance(origin, str):
        raise ValueError(f"origin must be a string, got {type(origin)}")

    global _ADDITIONAL_ORIGINS
    _ADDITIONAL_ORIGINS[_register_origin(origin)] = klass


def eqs_array_to_python_wrapper(eqs_array):
    """
    Convert a raw eqs_array to the corresponding Python ``ArrayWrapper``
    instance.
    """
    origin = data_origin(eqs_array)
    if _is_python_origin(origin):
        return _object_from_ptr(eqs_array.ptr)
    else:
        raise ValueError(
            f"unable to handle data coming from '{data_origin_name(origin)}'"
        )


def eqs_array_to_python_array(eqs_array, parent=None):
    """Convert a raw eqs_array to a Python ``Array``.

    Either the underlying array was allocated by Python, and the Python object
    is directly returned; or the underlying array was not allocated by Python,
    and additional origins are searched for a suitable Python wrapper class.
    """
    origin = data_origin(eqs_array)
    if _is_python_origin(origin):
        return _object_from_ptr(eqs_array.ptr).array
    elif origin in _ADDITIONAL_ORIGINS:
        return _ADDITIONAL_ORIGINS[origin](eqs_array, parent=parent)
    else:
        raise ValueError(
            f"unable to handle data coming from '{data_origin_name(origin)}'"
        )


def eqs_array_was_allocated_by_python(eqs_array):
    """Check if a given eqs_array was allocated by Python"""
    return _is_python_origin(data_origin(eqs_array))


def _is_python_origin(origin):
    return origin in [_origin_numpy(), _origin_pytorch()]


def data_origin(eqs_array):
    """Get the data origin of an eqs_array"""
    origin = eqs_data_origin_t()
    eqs_array.origin(eqs_array.ptr, origin)
    return origin.value


def data_origin_name(origin):
    """Get the name of the data origin of an eqs_array"""
    from .._c_lib import _get_library

    lib = _get_library()

    return _call_with_growing_buffer(
        lambda buffer, bufflen: lib.eqs_get_data_origin(origin, buffer, bufflen)
    )


# ============================================================================ #


class _RustNDArray(np.ndarray):
    """
    Small wrapper class around ``np.ndarray``, adding a reference to a parent
    Python object that actually owns the memory used inside the array. This
    prevents the parent from being garbage collected while the ndarray is still
    alive, thus preventing use-after-free.

    This is intended to be used to wrap Rust-owned memory inside a ndarray,
    while making sure the memory owner is kept around for long enough.

    This will be registered when loading the equistore library for the first time.
    """

    def __new__(cls, eqs_array, parent):
        from .._c_lib import _get_library

        lib = _get_library()

        data = ctypes.POINTER(ctypes.c_double)()
        shape_ptr = ctypes.POINTER(c_uintptr_t)()
        shape_count = c_uintptr_t()

        # this requires the data origin of this array is "rust.ndarray", which
        # is guaranteed by `eqs_array_to_python_object`. This function call will
        # fail anyway if this is not the case.
        lib.eqs_get_rust_array(eqs_array, data, shape_ptr, shape_count)

        shape = []
        for i in range(shape_count.value):
            shape.append(shape_ptr[i])

        array = _ptr_to_ndarray(data, shape, np.float64)
        obj = array.view(cls)

        # keep a reference to the parent object (if any) to prevent it from
        # being garbage-collected too early.
        obj._parent = parent

        return obj

    def __array_finalize__(self, obj):
        # keep the parent around when creating sub-views of this array
        self._parent = getattr(obj, "_parent", None)
