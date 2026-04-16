# -*- coding: utf-8 -*-
import atexit
import ctypes
import sys
from typing import Optional

from ._c_api import MTS_SUCCESS


class MetatensorError(Exception):
    """This class is used to throw exceptions for all errors in metatensor."""

    def __init__(self, message, status=None):
        super(Exception, self).__init__(message)

        self.message: str = message
        """error message for this exception"""

        self.status: Optional[int] = status
        """status code for this exception"""


def _check_status(status):
    if status == MTS_SUCCESS:
        return
    else:
        raise _get_exception(status)


def _check_pointer(pointer):
    if not pointer:
        raise _get_exception()


def _delete_exception(exception):
    # decrement the reference count of the exception
    exception_ptr = ctypes.cast(exception, ctypes.py_object).value
    ctypes.pythonapi.Py_DecRef(exception_ptr)


_DELETE_EXCEPTION = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(_delete_exception)


def _save_exception(e):
    """
    Save the given exception in metatensor's thread-local storage, so that it can be
    retrieved later with `_get_exception()`.
    """
    from ._c_lib import _get_library

    lib = _get_library()

    # increment the reference count of the exception
    exception_ptr = ctypes.py_object(e)
    ctypes.pythonapi.Py_IncRef(exception_ptr)

    try:
        lib.mts_set_last_error(
            ctypes.c_char_p(str(e).encode("utf8")),
            ctypes.c_char_p(b"Python exception"),
            ctypes.c_void_p.from_buffer(exception_ptr),
            _DELETE_EXCEPTION,
        )
    except Exception as err:
        # if we failed to save the exception, we are in a very bad state, but we should
        # still try to report the original error message if possible.
        print(
            "INTERNAL ERROR: unable to save last error after Python callback failure",
            file=sys.stderr,
        )
        print(
            f"original error was: {e}, error while saving was {err}",
            file=sys.stderr,
        )
        ctypes.pythonapi.Py_DecRef(exception_ptr)


def _get_exception(status=None):
    """
    Get the last error from libmetatensor that happened on the current thread.

    If the last error was caused by a Python exception, this returns the exception as
    is, otherwise it returns a new MetatensorError with the last error message.
    """
    from ._c_lib import _get_library

    lib = _get_library()
    message = ctypes.c_char_p()
    origin = ctypes.c_char_p()
    user_data = ctypes.c_void_p()
    status = lib.mts_last_error(
        ctypes.byref(message), ctypes.byref(origin), ctypes.byref(user_data)
    )

    if status != MTS_SUCCESS:
        return MetatensorError(
            "INTERNAL ERROR: failed to get the last error", status=status
        )

    if origin.value == b"Python exception" and user_data.value is not None:
        # This error was caused by a Python exception, so we re-raise it here
        # (the exception is stored in the user_data pointer)
        return ctypes.cast(user_data, ctypes.py_object).value

    return MetatensorError(message.value.decode("utf8"), status=status)


def _clear_last_error():
    """
    Clear the last error that happened if it was caused by a Python exception, to
    avoid trying to free the exception object after Python itself is unloaded.
    """
    from ._c_lib import _get_library

    try:
        lib = _get_library()
        origin = ctypes.c_char_p()
        status = lib.mts_last_error(None, ctypes.byref(origin), None)
        if status == MTS_SUCCESS and origin.value == b"Python exception":
            lib.mts_set_last_error(None, None, None, None)
    except Exception:
        pass


atexit.register(_clear_last_error)
