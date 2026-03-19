import ctypes
from typing import Union

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


@ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))
def _unversioned_wrapper_deleter(unversioned_ptr):
    """
    Deleter for an unversioned ``DLManagedTensor`` that wraps a
    ``DLManagedTensorVersioned``.

    Calls the versioned tensor's deleter (stored in ``manager_ctx``), then
    frees the unversioned wrapper.
    """
    if not unversioned_ptr:
        return

    versioned_ptr = ctypes.cast(
        unversioned_ptr.contents.manager_ctx,
        ctypes.POINTER(DLManagedTensorVersioned),
    )
    if versioned_ptr and versioned_ptr.contents.deleter:
        versioned_ptr.contents.deleter(versioned_ptr)

    PYTHON_API.PyMem_RawFree(ctypes.cast(unversioned_ptr, ctypes.c_void_p))


def wrap_versioned_as_unversioned(versioned_ptr):
    """
    Wrap a ``DLManagedTensorVersioned`` pointer inside a ``DLManagedTensor``.

    This is used for consumers (e.g. older PyTorch) that only understand
    the legacy ``"dltensor"`` capsule format.
    """
    unversioned_size = ctypes.sizeof(DLManagedTensor)
    unversioned_mem = PYTHON_API.PyMem_RawMalloc(unversioned_size)
    if not unversioned_mem:
        raise MemoryError("Failed to allocate DLManagedTensor")

    unversioned_ptr_out = ctypes.cast(unversioned_mem, ctypes.POINTER(DLManagedTensor))

    # Copy DLTensor from the versioned struct
    ctypes.memmove(
        ctypes.addressof(unversioned_ptr_out.contents.dl_tensor),
        ctypes.addressof(versioned_ptr.contents.dl_tensor),
        ctypes.sizeof(DLTensor),
    )

    # Store the original versioned pointer so the deleter can free it
    unversioned_ptr_out.contents.manager_ctx = ctypes.cast(
        versioned_ptr, ctypes.c_void_p
    )
    unversioned_ptr_out.contents.deleter = _unversioned_wrapper_deleter

    return unversioned_ptr_out


# ============================================================================ #
# PyCapsule creation for DLPack export
# ============================================================================ #

PYTHON_API.PyCapsule_New.restype = ctypes.py_object
PYTHON_API.PyCapsule_New.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
]


def make_dlpack_versioned_capsule(dl_managed_versioned_ptr):
    """
    Create a PyCapsule wrapping a ``DLManagedTensorVersioned`` pointer.

    The returned capsule can be passed to ``np.from_dlpack()`` or
    ``torch.from_dlpack()``.  The consumer is expected to take ownership
    (renaming the capsule to ``"used_dltensor_versioned"``).
    """
    ptr_value = ctypes.cast(dl_managed_versioned_ptr, ctypes.c_void_p).value
    if not ptr_value:
        raise ValueError("DLManagedTensorVersioned pointer is null")
    return PYTHON_API.PyCapsule_New(ptr_value, DLPACK_VERSIONED_NAME, None)


def make_dlpack_unversioned_capsule(dl_managed_ptr):
    """
    Create a PyCapsule wrapping a ``DLManagedTensor`` pointer.

    The returned capsule uses the legacy ``"dltensor"`` name for
    compatibility with consumers that do not support DLPack >= 1.0.
    """
    ptr_value = ctypes.cast(dl_managed_ptr, ctypes.c_void_p).value
    if not ptr_value:
        raise ValueError("DLManagedTensor pointer is null")
    return PYTHON_API.PyCapsule_New(ptr_value, DLPACK_NAME, None)


class DLPackArray:
    """
    A wrapper for raw DLPack pointers (i.e. not even a PyCapsule), implementing the
    ``__dlpack__`` protocol to allow being consumed by libraries like NumPy or PyTorch
    via DLPack.
    """

    def __init__(
        self,
        pointer: Union[
            ctypes.POINTER(DLManagedTensorVersioned),
            ctypes.POINTER(DLManagedTensor),
        ],
    ):
        self._pointer = pointer
        if isinstance(pointer, ctypes.POINTER(DLManagedTensorVersioned)):
            self._versioned = True
        elif isinstance(pointer, ctypes.POINTER(DLManagedTensor)):
            self._versioned = False
        else:
            raise TypeError(
                "pointer must be a ctypes pointer to either DLManagedTensorVersioned "
                "or DLManagedTensor"
            )

    def get(self):
        if self._pointer is not None:
            pointer = self._pointer
            self._pointer = None
            return pointer
        else:
            raise RuntimeError("DLPack tensor has already been consumed")

    def __del__(self):
        if self._pointer is not None:
            if self._pointer[0].deleter is not None:
                self._pointer[0].deleter(self._pointer)
                self._pointer = None

    def __dlpack__(
        self,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ):
        pointer = self._pointer
        if pointer is None:
            raise RuntimeError("can not call __dlpack__ twice on DLPackArray")

        if stream is not None:
            raise RuntimeError("only `stream=None` is supported")

        if self._versioned:
            version = pointer[0].version
            version = (version.major, version.minor)
            if max_version is not None:
                if version[0] > max_version[0]:
                    raise RuntimeError(
                        f"requested DLPack version {max_version}, but tensor has "
                        f"version {version}"
                    )

        if dl_device is not None and dl_device != self.__dlpack_device__():
            raise RuntimeError("device conversion is not supported")

        if copy is not None:
            raise RuntimeError("only `copy=None` is supported")

        self._pointer = None
        if self._versioned:
            capsule = make_dlpack_versioned_capsule(pointer)
        else:
            capsule = make_dlpack_unversioned_capsule(pointer)

        return capsule

    def __dlpack_device__(self):
        device = self._pointer[0].dl_tensor.device
        return (device.device_type, device.device_id)
