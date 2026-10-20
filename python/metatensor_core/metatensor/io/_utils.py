import ctypes
from typing import Callable

import numpy as np
from ctypes_dlpack import DLDataType, DLDataTypeCode

from .._c_api import c_uintptr_t, mts_array_t
from .._data._array import create_mts_array
from .._status import catch_exceptions, save_exception


def _resize_array(array, new_size):
    ctypes.resize(array, ctypes.sizeof(array._type_) * new_size)
    return (array._type_ * new_size).from_address(ctypes.addressof(array))


def _save_buffer_raw(mts_function, data) -> ctypes.Array:
    def realloc(buffer, _ptr, new_size):
        try:
            # convert void* to PyObject* and dereference to get a PyObject
            buffer = ctypes.cast(buffer, ctypes.POINTER(ctypes.py_object))
            buffer = buffer.contents.value

            # resize the buffer to grow it
            buffer = _resize_array(buffer, new_size)

            return ctypes.addressof(buffer)
        except Exception as e:
            # we don't want to propagate exceptions through C, so we catch anything
            # here, save the error and return a NULL pointer
            error = RuntimeError("failed to allocate more memory in realloc")
            error.__cause__ = e
            save_exception(error)
            return None

    # start with a buffer of 128 bytes in a ctypes string buffer (i.e. array of c_char)
    # we will be able to resize the allocation in `realloc` above, but the type will
    # stay `array of 128 c_char elements`.
    buffer = ctypes.create_string_buffer(128)

    # store the initial pointer and buffer_size on the stack, they will be modified by
    # `mts_tensormap_save_buffer`
    buffer_ptr = ctypes.c_char_p(ctypes.addressof(buffer))
    buffer_size = c_uintptr_t(buffer._length_)

    realloc_type = ctypes.CFUNCTYPE(
        ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p, c_uintptr_t
    )

    mts_function(
        buffer_ptr,
        buffer_size,
        # convert PyObject to void* to pass it to realloc
        ctypes.cast(ctypes.pointer(ctypes.py_object(buffer)), ctypes.c_void_p),
        realloc_type(realloc),
        data,
    )

    # remove extra data from the buffer, resizing it to the number of written bytes
    # (stored in buffer_size by the mts_function). The original buffer was already
    # resized by realloc during the C call; we create a view with the correct type
    # metadata and keep the original buffer alive by attaching it to the result.
    n = buffer_size.value
    ctypes.resize(buffer, n)
    result = (ctypes.c_char * n).from_address(ctypes.addressof(buffer))
    result._keepalive = buffer
    return result


# TODO: use a proper type alias when we drop support for Python <3.10; and remove the
# quotes around the type annotations using this.
# https://stackoverflow.com/a/73223518/4692076
CreateArrayCallback = Callable[
    [ctypes.POINTER(c_uintptr_t), c_uintptr_t, DLDataType, ctypes.POINTER(mts_array_t)],
    None,
]


_DLPACK_TO_NUMPY = {
    (DLDataTypeCode.kDLFloat, 16): np.float16,
    (DLDataTypeCode.kDLFloat, 32): np.float32,
    (DLDataTypeCode.kDLFloat, 64): np.float64,
    (DLDataTypeCode.kDLInt, 8): np.int8,
    (DLDataTypeCode.kDLInt, 16): np.int16,
    (DLDataTypeCode.kDLInt, 32): np.int32,
    (DLDataTypeCode.kDLInt, 64): np.int64,
    (DLDataTypeCode.kDLUInt, 8): np.uint8,
    (DLDataTypeCode.kDLUInt, 16): np.uint16,
    (DLDataTypeCode.kDLUInt, 32): np.uint32,
    (DLDataTypeCode.kDLUInt, 64): np.uint64,
    (DLDataTypeCode.kDLBool, 8): np.bool_,
    (DLDataTypeCode.kDLComplex, 64): np.complex64,
    (DLDataTypeCode.kDLComplex, 128): np.complex128,
}


def _dlpack_dtype_to_numpy(dtype):
    """Convert a DLDataType to a numpy dtype."""
    if dtype.lanes != 1:
        raise ValueError(f"unsupported DLDataType: lanes={dtype.lanes} (expected 1)")
    result = _DLPACK_TO_NUMPY.get((dtype.code, dtype.bits))
    if result is None:
        raise ValueError(
            f"unsupported DLDataType: code={dtype.code}, bits={dtype.bits}"
        )
    return result


def _dlpack_dtype_to_torch(dtype):
    """Convert a DLDataType to a torch dtype."""
    if dtype.lanes != 1:
        raise ValueError(
            f"unsupported DLDataType for torch: lanes={dtype.lanes} (expected 1)"
        )
    import torch

    _MAP = {
        (DLDataTypeCode.kDLFloat, 16): torch.float16,
        (DLDataTypeCode.kDLFloat, 32): torch.float32,
        (DLDataTypeCode.kDLFloat, 64): torch.float64,
        (DLDataTypeCode.kDLInt, 8): torch.int8,
        (DLDataTypeCode.kDLInt, 16): torch.int16,
        (DLDataTypeCode.kDLInt, 32): torch.int32,
        (DLDataTypeCode.kDLInt, 64): torch.int64,
        (DLDataTypeCode.kDLUInt, 8): torch.uint8,
        (DLDataTypeCode.kDLBool, 8): torch.bool,
        (DLDataTypeCode.kDLComplex, 64): torch.complex64,
        (DLDataTypeCode.kDLComplex, 128): torch.complex128,
    }
    result = _MAP.get((dtype.code, dtype.bits))
    if result is None:
        raise ValueError(
            f"unsupported DLDataType for torch: code={dtype.code}, bits={dtype.bits}"
        )
    return result


@catch_exceptions
def create_numpy_array(shape_ptr, shape_count, dtype, array):
    """
    Callback function that can be used with
    :py:func:`metatensor.io.load_custom_array` to load data in numpy arrays.
    """
    shape = []
    for i in range(shape_count):
        shape.append(shape_ptr[i])

    np_dtype = _dlpack_dtype_to_numpy(dtype)
    data = np.empty(shape, dtype=np_dtype)
    array[0] = create_mts_array(data)


@catch_exceptions
def create_torch_array(shape_ptr, shape_count, dtype, array):
    """
    Callback function that can be used with
    :py:func:`metatensor.io.load_custom_array` to load data in torch
    tensors. The resulting tensors are stored on CPU.
    """
    import torch

    shape = []
    for i in range(shape_count):
        shape.append(shape_ptr[i])

    torch_dtype = _dlpack_dtype_to_torch(dtype)
    data = torch.empty(shape, dtype=torch_dtype, device="cpu")
    array[0] = create_mts_array(data)
