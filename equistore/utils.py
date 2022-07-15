import ctypes
import operator
from functools import reduce

import numpy as np

from ._c_api import EQS_BUFFER_SIZE_ERROR
from .status import EquistoreError, _save_exception


def _call_with_growing_buffer(callback, initial=1024):
    bufflen = initial

    while True:
        buffer = ctypes.create_string_buffer(bufflen)
        try:
            callback(buffer, bufflen)
            break
        except EquistoreError as e:
            if e.status == EQS_BUFFER_SIZE_ERROR:
                # grow the buffer and retry
                bufflen *= 2
            else:
                raise
    return buffer.value.decode("utf8")


def catch_exceptions(function):
    def inner(*args, **kwargs):
        try:
            function(*args, **kwargs)
        except Exception as e:
            _save_exception(e)
            return -1
        return 0

    return inner


def _ptr_to_ndarray(ptr, shape, dtype):
    if reduce(operator.mul, shape) == 0:
        return np.empty(shape=shape, dtype=dtype)

    assert ptr is not None
    array = np.ctypeslib.as_array(ptr, shape=shape)
    assert array.dtype == dtype
    assert not array.flags["OWNDATA"]
    array.flags["WRITEABLE"] = True
    return array


def _ptr_to_const_ndarray(ptr, shape, dtype):
    array = _ptr_to_ndarray(ptr, shape, dtype)
    array.flags["WRITEABLE"] = False
    return array
