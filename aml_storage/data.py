import numpy as np
import ctypes
from typing import Union, NewType


from ._c_lib import _get_library
from ._c_api import aml_array_t, aml_data_origin_t, c_uintptr_t

from .utils import _call_with_growing_buffer, catch_exceptions

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


_NUMPY_STORAGE_ORIGIN = None
_TORCH_STORAGE_ORIGIN = None

if HAS_TORCH:
    # This NewType is only used for typechecking and documentation purposes
    Array = NewType("Array", Union[np.ndarray, torch.Tensor])
    """
    An ``Array`` contains the actual data stored in a
    :py:class:`aml_storage.Block`. This data is manipulated by ``aml_storage``
    in a completely opaque way: this library does not know what's inside the
    arrays appart from a small set of constrains:

    - array contains numeric data;
    - they are stored as row-major, n-dimensional arrays with at least 2 dimensions;
    - it is possible to create new arrays and move data from one array to
      another.

    The actual type of an ``Array`` depends on how the
    :py:class:`aml_storage.Block` was created. Currently, numpy ``ndarray`` and
    torch ``Tensor`` are supported.
    """

else:
    Array = NewType("Array", np.ndarray)


def _is_numpy_array(array):
    return isinstance(array, np.ndarray)


def _numpy_origin():
    global _NUMPY_STORAGE_ORIGIN
    if _NUMPY_STORAGE_ORIGIN is None:
        _NUMPY_STORAGE_ORIGIN = ctypes.c_uint64(0)
        lib = _get_library()

        origin_name = __name__ + ".numpy"
        lib.aml_register_data_origin(origin_name.encode("utf8"), _NUMPY_STORAGE_ORIGIN)
        _NUMPY_STORAGE_ORIGIN = _NUMPY_STORAGE_ORIGIN.value

    return _NUMPY_STORAGE_ORIGIN


def _is_torch_array(array):
    if not HAS_TORCH:
        return False

    return isinstance(array, torch.Tensor)


def _torch_origin():
    global _TORCH_STORAGE_ORIGIN
    if _TORCH_STORAGE_ORIGIN is None:
        _TORCH_STORAGE_ORIGIN = ctypes.c_uint64(0)
        lib = _get_library()

        origin_name = __name__ + ".torch"
        lib.aml_register_data_origin(origin_name.encode("utf8"), _TORCH_STORAGE_ORIGIN)
        _TORCH_STORAGE_ORIGIN = _TORCH_STORAGE_ORIGIN.value

    return _TORCH_STORAGE_ORIGIN


def aml_array_to_python_object(data):
    origin = data_origin(data)
    if origin in [_NUMPY_STORAGE_ORIGIN, _TORCH_STORAGE_ORIGIN]:
        return _object_from_ptr(data.ptr)
    else:
        raise ValueError(
            f"unable to handle data coming from '{data_origin_name(origin)}'"
        )


def data_origin(aml_array):
    origin = aml_data_origin_t()
    aml_array.origin(aml_array.ptr, origin)
    return origin.value


def data_origin_name(origin):
    lib = _get_library()

    return _call_with_growing_buffer(
        lambda buffer, bufflen: lib.aml_get_data_origin(origin, buffer, bufflen)
    )


class AmlData:
    """
    Small wrapper making Python arrays compatible with ``aml_array_t``.
    """

    def __init__(self, array):
        self.array = array
        self._shape = ctypes.ARRAY(c_uintptr_t, len(array.shape))(*array.shape)

        self._children = []
        if _is_numpy_array(array):
            array_origin = _numpy_origin()
        elif _is_torch_array(array):
            array_origin = _torch_origin()
        else:
            raise ValueError(f"unknown array type: {type(array)}")

        self.aml_array = aml_array_t()

        # `aml_data.aml_array.ptr` is a pointer to the PyObject `self`
        self.aml_array.ptr = ctypes.cast(
            ctypes.pointer(ctypes.py_object(self)), ctypes.c_void_p
        )

        @catch_exceptions
        def aml_storage_origin(this, origin):
            origin[0] = array_origin

        # use storage.XXX.__class__ to get the right type for all functions
        self.aml_array.origin = self.aml_array.origin.__class__(aml_storage_origin)

        self.aml_array.shape = self.aml_array.shape.__class__(_aml_storage_shape)
        self.aml_array.reshape = self.aml_array.reshape.__class__(_aml_storage_reshape)
        self.aml_array.swap_axes = self.aml_array.swap_axes.__class__(
            _aml_storage_swap_axes
        )

        self.aml_array.create = self.aml_array.create.__class__(_aml_storage_create)
        self.aml_array.copy = self.aml_array.copy.__class__(_aml_storage_copy)
        self.aml_array.destroy = self.aml_array.destroy.__class__(_aml_storage_destroy)

        self.aml_array.move_sample = self.aml_array.move_sample.__class__(
            _aml_storage_move_sample
        )


def _object_from_ptr(ptr):
    """Extract the Python object from a pointer to the PyObject"""
    return ctypes.cast(ptr, ctypes.POINTER(ctypes.py_object)).contents.value


@catch_exceptions
def _aml_storage_shape(this, shape_ptr, shape_count):
    storage = _object_from_ptr(this)

    shape_ptr[0] = storage._shape
    shape_count[0] = len(storage._shape)


@catch_exceptions
def _aml_storage_reshape(this, shape_ptr, shape_count):
    storage = _object_from_ptr(this)

    shape = []
    for i in range(shape_count):
        shape.append(shape_ptr[i])

    storage.array = storage.array.reshape(shape)
    storage._shape = ctypes.ARRAY(c_uintptr_t, len(shape))(*shape)


@catch_exceptions
def _aml_storage_swap_axes(this, axis_1, axis_2):
    storage = _object_from_ptr(this)
    storage.array = storage.array.swapaxes(axis_1, axis_2)

    shape = storage.array.shape
    storage._shape = ctypes.ARRAY(c_uintptr_t, len(shape))(*shape)


@catch_exceptions
def _aml_storage_create(this, shape_ptr, shape_count, data_storage):
    storage = _object_from_ptr(this)

    shape = []
    for i in range(shape_count):
        shape.append(shape_ptr[i])
    dtype = storage.array.dtype

    if _is_numpy_array(storage.array):
        array = np.zeros(shape, dtype=dtype)
    elif _is_torch_array(storage.array):
        array = torch.zeros(shape, dtype=dtype, device=storage.array.device)

    array = AmlData(array)
    storage._children.append(array)

    data_storage[0] = array.aml_array


@catch_exceptions
def _aml_storage_copy(this, data_storage):
    storage = _object_from_ptr(this)

    if _is_numpy_array(storage.array):
        array = storage.array.copy()
    elif _is_torch_array(storage.array):
        array = storage.array.clone()

    array = AmlData(array)
    data_storage[0] = array.aml_array


@catch_exceptions
def _aml_storage_destroy(this):
    storage = _object_from_ptr(this)
    storage.array = None


@catch_exceptions
def _aml_storage_move_sample(
    this, sample, feature_start, feature_stop, other, other_sample
):
    other = _object_from_ptr(other).array
    output = _object_from_ptr(this).array
    output[sample, ..., feature_start:feature_stop] = other[other_sample, ..., :]
