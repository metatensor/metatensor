import ctypes

from .._c_api import (
    DLManagedTensorVersioned,
    DLPackVersion,
    DLTensor,
)


# ============================================================================ #
# DLPack C API handling
# ============================================================================ #

PYTHON_API = ctypes.pythonapi

# Define C-API signatures
PYTHON_API.PyCapsule_GetPointer.restype = ctypes.c_void_p
PYTHON_API.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

PYTHON_API.PyCapsule_SetName.restype = ctypes.c_int
PYTHON_API.PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]

PYTHON_API.PyCapsule_GetName.restype = ctypes.c_char_p
PYTHON_API.PyCapsule_GetName.argtypes = [ctypes.py_object]

PYTHON_API.PyErr_Occurred.restype = ctypes.c_void_p
PYTHON_API.PyErr_Occurred.argtypes = []

PYTHON_API.PyErr_Clear.restype = None
PYTHON_API.PyErr_Clear.argtypes = []

PYTHON_API.PyMem_RawMalloc.restype = ctypes.c_void_p
PYTHON_API.PyMem_RawMalloc.argtypes = [ctypes.c_size_t]

PYTHON_API.PyMem_RawFree.restype = None
PYTHON_API.PyMem_RawFree.argtypes = [ctypes.c_void_p]

PYTHON_API.Py_IncRef.restype = None
PYTHON_API.Py_IncRef.argtypes = [ctypes.py_object]

PYTHON_API.Py_DecRef.restype = None
PYTHON_API.Py_DecRef.argtypes = [ctypes.py_object]

# DLPack capsule names
DLPACK_VERSIONED_NAME = b"dltensor_versioned"
USED_DLPACK_VERSIONED_NAME = b"used_dltensor_versioned"
DLPACK_NAME = b"dltensor"
USED_DLPACK_NAME = b"used_dltensor"


class DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))

DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLManagedTensorDeleter),
]


@ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensorVersioned))
def _versioned_wrapper_deleter(versioned_ptr):
    """
    This function runs when a ``DLManagedTensorVersioned`` object that wraps a
    ``DLManagedTensor`` is freed.

    The legacy tensor without version has the deleter called explicitly, and then we
    free the versioned wrapper itself.
    """
    if not versioned_ptr:
        return

    versioned = versioned_ptr.contents
    original_ptr = ctypes.cast(versioned.manager_ctx, ctypes.POINTER(DLManagedTensor))

    if original_ptr and original_ptr.contents.deleter:
        original_ptr.contents.deleter(original_ptr)

    # Free the versioned wrapper struct
    PYTHON_API.PyMem_RawFree(ctypes.cast(versioned_ptr, ctypes.c_void_p))


def wrap_unversioned_as_versioned(unversioned_ptr):
    """
    Wrap a pointer to ``DLManagedTensor`` inside a ``DLManagedTensorVersioned``.

    This is used to convert legacy DLPack tensors to the versioned format.
    """

    # Allocate Versioned Struct
    versioned_size = ctypes.sizeof(DLManagedTensorVersioned)
    versioned_mem = PYTHON_API.PyMem_RawMalloc(versioned_size)
    if not versioned_mem:
        raise MemoryError("Failed to allocate DLManagedTensorVersioned")

    versioned_ptr = ctypes.cast(versioned_mem, ctypes.POINTER(DLManagedTensorVersioned))
    versioned = versioned_ptr.contents

    # Populate Versioned Struct
    versioned.version = DLPackVersion()
    versioned.version.major = 1
    versioned.version.minor = 0
    # use the context to store the original unversioned tensor pointer
    versioned.manager_ctx = ctypes.cast(unversioned_ptr, ctypes.c_void_p)
    versioned.deleter = _versioned_wrapper_deleter
    versioned.flags = 0

    # Copy DLTensor from the `unversioned_ptr`. The data is the first member of the
    # struct
    ctypes.memmove(
        ctypes.addressof(versioned.dl_tensor),
        unversioned_ptr,
        ctypes.sizeof(DLTensor),
    )

    return versioned_ptr
