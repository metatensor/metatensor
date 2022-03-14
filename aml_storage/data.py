import numpy as np
import ctypes


from ._c_lib import _get_library
from ._c_api import aml_array_t, aml_data_origin_t

from .utils import _call_with_growing_buffer, catch_exceptions

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


_NUMPY_STORAGE_ORIGIN = None
_TORCH_STORAGE_ORIGIN = None


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


def aml_data_to_array(data):
    origin = _data_origin(data[0]).value
    if origin == _NUMPY_STORAGE_ORIGIN:
        return _object_from_ptr(data[0].ptr).array
    elif origin == _TORCH_STORAGE_ORIGIN:
        return _object_from_ptr(data[0].ptr).array
    else:
        raise ValueError(
            f"unable to handle data coming from '{_data_origin_name(origin)}'"
        )


def _data_origin(data):
    origin = aml_data_origin_t()
    data.origin(data.ptr, origin)
    return origin


def _data_origin_name(origin):
    lib = _get_library()

    return _call_with_growing_buffer(
        lambda buffer, bufflen: lib.aml_get_data_origin(origin, buffer, bufflen)
    )


class AmlData:
    def __init__(self, array):
        self.array = array
        self._children = []
        if _is_numpy_array(array):
            array_origin = _numpy_origin()
        elif _is_torch_array(array):
            array_origin = _torch_origin()
        else:
            raise ValueError(f"unknown array type: {type(array)}")

        self._storage = aml_array_t()

        # `aml_data_storage.ptr` is a pointer to the PyObject `self`
        self._storage.ptr = ctypes.cast(
            ctypes.pointer(ctypes.py_object(self)), ctypes.c_void_p
        )

        @catch_exceptions
        def aml_storage_origin(this, origin):
            origin[0] = array_origin

        # use storage.XXX.__class__ to get the right type for all functions
        self._storage.origin = self._storage.origin.__class__(aml_storage_origin)

        self._storage.shape = self._storage.shape.__class__(_aml_storage_shape)
        self._storage.reshape = self._storage.reshape.__class__(_aml_storage_reshape)

        self._storage.create = self._storage.create.__class__(_aml_storage_create)
        self._storage.destroy = self._storage.destroy.__class__(_aml_storage_destroy)

        self._storage.set_from = self._storage.set_from.__class__(_aml_storage_set_from)


def _object_from_ptr(ptr):
    """Extract the Python object from a pointer to the PyObject"""
    return ctypes.cast(ptr, ctypes.POINTER(ctypes.py_object)).contents.value


@catch_exceptions
def _aml_storage_shape(this, n_samples, n_components, n_features):
    storage = _object_from_ptr(this)

    shape = storage.array.shape

    n_samples[0] = ctypes.c_uint64(shape[0])
    n_components[0] = ctypes.c_uint64(shape[1])
    n_features[0] = ctypes.c_uint64(shape[2])


@catch_exceptions
def _aml_storage_reshape(this, n_samples, n_components, n_features):
    storage = _object_from_ptr(this)
    storage.array = storage.array.reshape((n_samples, n_components, n_features))


@catch_exceptions
def _aml_storage_create(this, n_samples, n_components, n_features, data_storage):
    storage = _object_from_ptr(this)

    shape = (n_samples, n_components, n_features)
    dtype = storage.array.dtype

    if _is_numpy_array(storage.array):
        array = np.zeros(shape, dtype=dtype)
    elif _is_torch_array(storage.array):
        array = torch.zeros(shape, dtype=dtype, device=storage.array.device)

    array = AmlData(array)
    storage._children.append(array)

    data_storage[0] = array._storage


@catch_exceptions
def _aml_storage_destroy(this):
    storage = _object_from_ptr(this)
    storage.array = None


@catch_exceptions
def _aml_storage_set_from(
    this, sample, feature_start, feature_stop, other, other_sample
):
    other = _object_from_ptr(other).array
    output = _object_from_ptr(this).array
    output[sample, :, feature_start:feature_stop] = other[other_sample, :, :]
