# -*- coding: utf-8 -*-
from typing import Optional

from ._c_api import AML_SUCCESS
from ._c_lib import _get_library


class AmlError(Exception):
    """
    This class is used to throw exceptions for all errors in AML.
    """

    def __init__(self, message, status=None):
        super(Exception, self).__init__(message)

        self.message: str = message
        """error message for this exception"""

        self.status: Optional[int] = status
        """status code for this exception"""


LAST_EXCEPTION = None


def _save_exception(e):
    global LAST_EXCEPTION
    LAST_EXCEPTION = e


def _check_status(status):
    if status == AML_SUCCESS:
        return
    elif status > AML_SUCCESS:
        raise AmlError(last_error(), status)
    elif status < AML_SUCCESS:
        global LAST_EXCEPTION
        e = LAST_EXCEPTION
        LAST_EXCEPTION = None
        raise AmlError(last_error(), status) from e


def _check_pointer(pointer):
    if not pointer:
        global LAST_EXCEPTION
        if LAST_EXCEPTION is not None:
            e = LAST_EXCEPTION
            LAST_EXCEPTION = None
            raise AmlError(last_error()) from e
        else:
            raise AmlError(last_error())


def last_error():
    """Get the last error message on this thread"""
    lib = _get_library()
    message = lib.aml_last_error()
    return message.decode("utf8")
