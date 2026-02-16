import ctypes

from .._c_api import c_uintptr_t
from ..status import _save_exception


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
            _save_exception(error)
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
    # (stored in buffer_size by the mts_function)
    buffer = _resize_array(buffer, buffer_size.value)

    return buffer
