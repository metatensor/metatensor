import ctypes
from typing import Union

import numpy as np

from .._c_api import c_uintptr_t, mts_array_t, mts_data_origin_t
from ..utils import catch_exceptions


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _register_origin(name):
    from .._c_lib import _get_library

    lib = _get_library()
    origin = mts_data_origin_t(0)
    lib.mts_register_data_origin(name.encode("utf8"), origin)
    return origin.value


def _is_numpy_array(array):
    return isinstance(array, np.ndarray)


def _is_torch_array(array):
    if not HAS_TORCH:
        return False

    return isinstance(array, torch.Tensor)


_NUMPY_STORAGE_ORIGIN = None
_TORCH_STORAGE_ORIGIN = None


def _origin_numpy():
    global _NUMPY_STORAGE_ORIGIN
    if _NUMPY_STORAGE_ORIGIN is None:
        _NUMPY_STORAGE_ORIGIN = _register_origin(__name__ + ".numpy")

    return _NUMPY_STORAGE_ORIGIN


def _origin_pytorch():
    global _TORCH_STORAGE_ORIGIN
    if _TORCH_STORAGE_ORIGIN is None:
        _TORCH_STORAGE_ORIGIN = _register_origin(__name__ + ".torch")

    return _TORCH_STORAGE_ORIGIN


if HAS_TORCH:
    torch_dtype = torch.dtype
    torch_device = torch.device
else:

    class torch_dtype:
        pass

    class torch_device:
        pass


DType = Union[np.dtype, torch_dtype]
"""Type representing a dtype in either numpy or torch"""

Device = Union[str, torch_device]
"""Type representing a device in either numpy or torch"""


def array_dtype(array) -> DType:
    """Get the dtype of an array"""
    if _is_numpy_array(array) or _is_torch_array(array):
        return array.dtype
    else:
        raise TypeError(f"unknown array type: {type(array)}")


def array_change_dtype(array, dtype: DType, non_blocking: bool):
    """Change the dtype of an array"""
    if _is_numpy_array(array):
        return array.astype(dtype)
    elif _is_torch_array(array):
        return array.to(dtype=dtype, non_blocking=non_blocking)
    else:
        raise TypeError(f"unknown array type: {type(array)}")


def array_device(array) -> Device:
    """Get the device of an array"""
    if _is_numpy_array(array):
        return "cpu"
    elif _is_torch_array(array):
        return array.device
    else:
        raise TypeError(f"unknown array type: {type(array)}")


def array_device_is_cpu(array) -> bool:
    """Check if the device of an array is CPU"""
    if _is_numpy_array(array):
        return True
    elif _is_torch_array(array):
        return array.device.type == torch.device("cpu").type
    else:
        raise TypeError(f"unknown array type: {type(array)}")


def array_change_device(array, device: Device, non_blocking: bool):
    """Change the device of an array"""
    if _is_numpy_array(array):
        if device != "cpu":
            raise ValueError(f"can not move numpy array to non-cpu device: {device}")
        return array
    elif _is_torch_array(array):
        return array.to(device=device, non_blocking=non_blocking)
    else:
        raise TypeError(f"unknown array type: {type(array)}")


def array_change_backend(array, backend: str):
    if _is_numpy_array(array):
        if backend == "numpy":
            return array
        elif backend == "torch":
            if not HAS_TORCH:
                raise ModuleNotFoundError(
                    "can not convert to `torch` arrays since PyTorch is not installed"
                )
            else:
                return torch.from_numpy(array)
        else:
            raise ValueError(f"unknown array backend: '{backend}'")

    elif _is_torch_array(array):
        if backend == "numpy":
            return array.numpy()
        elif backend == "torch":
            return array
        else:
            raise ValueError(f"unknown array backend: '{backend}'")

    else:
        raise TypeError(f"unknown array type: {type(array)}")


class DeviceWarning(RuntimeWarning):
    """
    Custom warning class for device mismatch in :py:class:`TensorBlock` and
    :py:class:`TensorMap`.
    """


class ArrayWrapper:
    """Store together some Python array and it's shape as C-compatible data"""

    __slots__ = ["array", "c_shape"]

    def __init__(self, array, c_shape):
        self.array = array
        self.c_shape = c_shape


# We keep wrappers of Python arrays alive by storing a reference to them in this
# dictionary. Each value is stored under the `id(value)` key, which makes sure no two
# values have the same key.
_KNOWN_ARRAY_WRAPPERS = {}

# cache array types by array length, since creating the type everytime is a bit slow
_POSSIBLE_C_SHAPE_TYPES = [ctypes.ARRAY(c_uintptr_t, N) for N in range(64)]


def create_mts_array(array):
    """
    Create a ``mts_array_t`` corresponding to the given ``array``, which should be
    either :py:class:`torch.Tensor` or :py:class:`numpy.ndarray`.
    """
    c_shape = _POSSIBLE_C_SHAPE_TYPES[len(array.shape)]()
    c_shape[:] = array.shape

    wrapper = ArrayWrapper(array, c_shape)

    if _is_numpy_array(array):
        mts_array_origin = _MTS_ARRAY_ORIGIN_NUMPY
    elif _is_torch_array(array):
        mts_array_origin = _MTS_ARRAY_ORIGIN_PYTORCH
    else:
        raise ValueError(f"unknown array type: {type(array)}")

    global _KNOWN_ARRAY_WRAPPERS
    _KNOWN_ARRAY_WRAPPERS[id(wrapper)] = wrapper

    mts_array = mts_array_t.from_buffer_copy(_DEFAULT_MTS_ARRAY)

    # we use id(wrapper) as the C API user-data "pointer", since we can use it together
    # with `_KNOWN_ARRAY_WRAPPERS` to recover the corresponding Python object.
    mts_array.ptr = id(wrapper)
    mts_array.origin = mts_array_origin

    return mts_array


@catch_exceptions
def _mts_array_data(this, data):
    wrapper = _KNOWN_ARRAY_WRAPPERS[this]

    if _is_numpy_array(wrapper.array):
        array = wrapper.array

    elif _is_torch_array(wrapper.array):
        array = wrapper.array

        if array.device.type != "cpu":
            raise ValueError("can only get data pointer for tensors on CPU")

        # `.numpy()` will fail if the data is on GPU or requires gradient
        # tracking, and the resulting array is sharing data storage with the
        # tensor, meaning we can take a pointer to it without the array being
        # freed immediately.
        array = array.numpy()

    if not array.data.c_contiguous:
        raise ValueError("can not get data pointer for non contiguous array")

    if not array.dtype == np.float64:
        raise ValueError(
            f"can not get data pointer for array type {array.dtype}, "
            "only float64 is supported. If you are trying to save a TensorMap "
            "to a file, you can set `use_numpy=True`."
        )

    data[0] = array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


@catch_exceptions
def _mts_array_shape(this, shape_ptr, shape_count):
    wrapper = _KNOWN_ARRAY_WRAPPERS[this]

    shape_ptr[0] = wrapper.c_shape
    shape_count[0] = len(wrapper.c_shape)


@catch_exceptions
def _mts_array_reshape(this, shape_ptr, shape_count):
    wrapper = _KNOWN_ARRAY_WRAPPERS[this]

    shape = []
    for i in range(shape_count):
        shape.append(shape_ptr[i])

    wrapper.array = wrapper.array.reshape(shape)
    wrapper.c_shape = _POSSIBLE_C_SHAPE_TYPES[len(shape)]()
    wrapper.c_shape[:] = shape


@catch_exceptions
def _mts_array_swap_axes(this, axis_1, axis_2):
    wrapper = _KNOWN_ARRAY_WRAPPERS[this]
    wrapper.array = wrapper.array.swapaxes(axis_1, axis_2)

    shape = wrapper.array.shape
    wrapper.c_shape = _POSSIBLE_C_SHAPE_TYPES[len(shape)]()
    wrapper.c_shape[:] = shape


@catch_exceptions
def _mts_array_create(this, shape_ptr, shape_count, new_array):
    wrapper = _KNOWN_ARRAY_WRAPPERS[this]

    shape = []
    for i in range(shape_count):
        shape.append(shape_ptr[i])
    dtype = wrapper.array.dtype

    if _is_numpy_array(wrapper.array):
        array = np.zeros(shape, dtype=dtype)
    elif _is_torch_array(wrapper.array):
        array = torch.zeros(shape, dtype=dtype, device=wrapper.array.device)

    new_array[0] = create_mts_array(array)


@catch_exceptions
def _mts_array_copy(this, new_array):
    wrapper = _KNOWN_ARRAY_WRAPPERS[this]

    if _is_numpy_array(wrapper.array):
        array = wrapper.array.copy()
    elif _is_torch_array(wrapper.array):
        array = wrapper.array.clone()

    new_array[0] = create_mts_array(array)


@catch_exceptions
def _mts_array_destroy(this):
    del _KNOWN_ARRAY_WRAPPERS[this]


@catch_exceptions
def _mts_array_move_samples_from(
    this,
    input,
    samples_ptr,
    samples_count,
    property_start,
    property_end,
):
    output = _KNOWN_ARRAY_WRAPPERS[this].array
    input = _KNOWN_ARRAY_WRAPPERS[input].array

    input_samples = []
    output_samples = []
    for i in range(samples_count):
        input_samples.append(samples_ptr[i].input)
        output_samples.append(samples_ptr[i].output)

    properties = slice(property_start, property_end)
    output[output_samples, ..., properties] = input[input_samples, ..., :]


def _cast_to_ctype_functype(function, field_name):
    for name, klass in mts_array_t._fields_:
        if name == field_name:
            return klass(function)

    raise ValueError(f"no field named {field_name} in mts_array_t")


_MTS_ARRAY_DATA = _cast_to_ctype_functype(_mts_array_data, "data")
_MTS_ARRAY_SHAPE = _cast_to_ctype_functype(_mts_array_shape, "shape")
_MTS_ARRAY_RESHAPE = _cast_to_ctype_functype(_mts_array_reshape, "reshape")
_MTS_ARRAY_SWAP_AXES = _cast_to_ctype_functype(_mts_array_swap_axes, "swap_axes")
_MTS_ARRAY_CREATE = _cast_to_ctype_functype(_mts_array_create, "create")
_MTS_ARRAY_COPY = _cast_to_ctype_functype(_mts_array_copy, "copy")
_MTS_ARRAY_DESTROY = _cast_to_ctype_functype(_mts_array_destroy, "destroy")
_MTS_ARRAY_MOVE_SAMPLES_FROM = _cast_to_ctype_functype(
    _mts_array_move_samples_from, "move_samples_from"
)


@catch_exceptions
def mts_array_origin_numpy(this, origin):
    origin[0] = _origin_numpy()


@catch_exceptions
def mts_array_origin_pytorch(this, origin):
    origin[0] = _origin_pytorch()


_MTS_ARRAY_ORIGIN_NUMPY = _cast_to_ctype_functype(mts_array_origin_numpy, "origin")
_MTS_ARRAY_ORIGIN_PYTORCH = _cast_to_ctype_functype(mts_array_origin_pytorch, "origin")


# The default value for all Python-provided `mts_array_t`. Only the first two members
# will change, having a pre-allocated instance will make it faster to create new ones
# with `mts_array_t.from_buffer_copy`.
_DEFAULT_MTS_ARRAY = mts_array_t(
    ptr=0,
    origin=_cast_to_ctype_functype(lambda u: u, "origin"),
    data=_MTS_ARRAY_DATA,
    shape=_MTS_ARRAY_SHAPE,
    reshape=_MTS_ARRAY_RESHAPE,
    swap_axes=_MTS_ARRAY_SWAP_AXES,
    create=_MTS_ARRAY_CREATE,
    copy=_MTS_ARRAY_COPY,
    destroy=_MTS_ARRAY_DESTROY,
    move_samples_from=_MTS_ARRAY_MOVE_SAMPLES_FROM,
)
