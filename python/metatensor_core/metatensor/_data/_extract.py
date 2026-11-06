import ctypes
import importlib
import typing
from typing import Any, Literal

import numpy as np
from ctypes_dlpack import (
    DLDataType,
    DLDevice,
    DLDeviceType,
    DLManagedTensorVersioned,
    DLPackArray,
    DLPackVersion,
)

from .._c_api import (
    c_uintptr_t,
    mts_array_t,
    mts_data_origin_t,
    mts_status_t,
)
from .._status import MetatensorError, check_status
from ._array import (
    _KNOWN_ARRAY_WRAPPERS,
    _origin_numpy,
    _origin_pytorch,
    array_device,
    array_dtype,
)


ArrayBackend = Literal["numpy", "torch"]
"""
Array backend used to give access to data that was not allocated by Python. This can be
one of:

- ``"numpy"``: data is returned as :py:class:`numpy.ndarray`, and must live on CPU;
- ``"torch"``: data is returned as :py:class:`torch.Tensor`.
"""

_ARRAY_BACKENDS = typing.get_args(ArrayBackend)


def check_array_backend(array_backend: str) -> str:
    """Check that ``array_backend`` is one of the supported array backends"""
    if array_backend not in _ARRAY_BACKENDS:
        raise ValueError(
            f"invalid array_backend '{array_backend}', expected one of "
            + ", ".join(f"'{b}'" for b in _ARRAY_BACKENDS)
        )
    return array_backend


def register_external_data_wrapper(origin, klass):
    """
    .. deprecated:: 0.2.7

        This function does nothing. Data from any non-Python origin is now accessible
        through DLPack, use the ``array_backend`` parameter of
        :py:meth:`TensorMap.unsafe_from_ptr`, :py:meth:`TensorBlock.unsafe_from_ptr`,
        :py:meth:`Labels.unsafe_from_ptr` or the loading functions to choose which
        array type to use for this data.

    :param origin: data origin name as a string, corresponding to the output of
        :c:func:`mts_array_t.origin`
    :param klass: wrapper class to use for this origin
    """


def mts_array_to_python_array(mts_array, parent=None, array_backend="numpy"):
    """Convert a raw mts_array to a Python ``Array``.

    Either the underlying array was allocated by Python, and the Python object is
    directly returned; or the underlying array was not allocated by Python, and the data
    is exported through DLPack and given to the requested ``array_backend``.

    :param mts_array: raw array to convert
    :param parent: Python object owning ``mts_array``, which will be kept alive for as
        long as the returned array is alive. This should not hold a reference to the
        returned array, or the corresponding memory will never be released.
    :param array_backend: array backend to use for non-Python data
    """
    origin = data_origin(mts_array)
    if _is_python_origin(origin):
        return _KNOWN_ARRAY_WRAPPERS[mts_array.ptr].array

    check_array_backend(array_backend)
    if array_backend == "torch":
        backend_module = import_array_backend(array_backend)

    device = mts_array_device(mts_array)
    if array_backend == "numpy" and device.device_type != DLDeviceType.kDLCPU:
        raise ValueError(
            f"can not access data on device '{device.device_type}:{device.device_id}' "
            "with array_backend='numpy', use array_backend='torch' instead"
        )

    array = mts_array_to_dlpack(mts_array, parent=parent)
    if array_backend == "numpy":
        return np.from_dlpack(array)
    else:
        return backend_module.from_dlpack(array)


def mts_array_to_dlpack(mts_array, parent=None, dl_device=None) -> DLPackArray:
    """
    Export the data in a raw mts_array through DLPack, regardless of where the data was
    allocated.

    :param mts_array: raw array to export
    :param parent: Python object owning ``mts_array``, which will be kept alive for as
        long as the returned array is alive. This should not hold a reference to the
        returned array, or the corresponding memory will never be released.
    :param dl_device: device on which the data should be exported, as a ``(device_type,
        device_id)`` tuple following the DLPack convention. If ``None``, the data is
        exported on the device where it currently lives.
    """
    if dl_device is None:
        device = mts_array_device(mts_array)
    else:
        device_type, device_id = dl_device
        device = DLDevice(device_type=device_type, device_id=device_id)

    dl_managed_ptr = ctypes.POINTER(DLManagedTensorVersioned)()
    status = mts_array.as_dlpack(
        mts_array.ptr,
        ctypes.byref(dl_managed_ptr),
        device,
        None,
        DLPackVersion(major=1, minor=0),
    )
    check_status(status)

    return DLPackArray(dl_managed_ptr, keepalive=parent)


def mts_array_dtype(mts_array, array_backend=None):
    """
    Get the dtype of an mts_array. If ``array_backend`` is ``None``, this returns the
    raw ``DLDataType``. Otherwise, this returns the dtype as it would be returned by
    :py:func:`mts_array_to_python_array` with the same ``array_backend``.
    """
    if array_backend is None:
        dtype = DLDataType()
        check_status(mts_array.dtype(mts_array.ptr, ctypes.byref(dtype)))
        return dtype

    array = mts_array_to_python_array(mts_array, array_backend=array_backend)
    return array_dtype(array)


def mts_array_device(mts_array, array_backend=None):
    """
    Get the device of an mts_array. If ``array_backend`` is ``None``, this returns the
    raw ``DLDevice``. Otherwise, this returns the device as it would be returned by
    :py:func:`mts_array_to_python_array` with the same ``array_backend``.
    """
    if array_backend is None:
        device = DLDevice()
        check_status(mts_array.device(mts_array.ptr, ctypes.byref(device)))
        return device

    array = mts_array_to_python_array(mts_array, array_backend=array_backend)
    return array_device(array)


def mts_array_shape(mts_array):
    """Get the shape of an mts_array as a tuple of integers"""
    shape_ptr = ctypes.POINTER(c_uintptr_t)()
    shape_count = c_uintptr_t()
    status = mts_array.shape(
        mts_array.ptr, ctypes.byref(shape_ptr), ctypes.byref(shape_count)
    )
    check_status(status)

    return tuple(shape_ptr[i] for i in range(shape_count.value))


def import_array_backend(name):
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise ImportError(
            f"array_backend='{name}' requires the {name} package, "
            f"install it with `pip install {name}`"
        ) from e


def mts_array_was_allocated_by_python(mts_array):
    """Check if a given mts_array was allocated by Python"""
    return _is_python_origin(data_origin(mts_array))


def _is_python_origin(origin):
    return origin in [_origin_numpy(), _origin_pytorch()]


def data_origin(mts_array):
    """Get the data origin of an mts_array"""
    origin = mts_data_origin_t()

    status = mts_array.origin(mts_array.ptr, origin)
    check_status(status)

    return origin.value


def data_origin_name(origin):
    """Get the name of the data origin of an mts_array"""
    from .._c_lib import _get_library

    lib = _get_library()

    return _call_with_growing_buffer(
        lambda buffer, bufflen: lib.mts_get_data_origin(origin, buffer, bufflen)
    )


# ============================================================================ #


class ExternalCpuArray(np.ndarray):
    """
    .. deprecated:: 0.2.7

        Data from non-Python origins is now returned as plain
        :py:class:`numpy.ndarray` by default, this class is no longer used by
        metatensor.

    Small wrapper class around ``np.ndarray``, adding a reference to a parent
    Python object that actually owns the memory used inside the array. This
    prevents the parent from being garbage collected while the ndarray is still
    alive, thus preventing use-after-free.

    This is intended to be used to wrap Rust-owned memory inside a numpy's
    ndarray, while making sure the memory owner is kept around for long enough.
    """

    def __new__(cls, mts_array: mts_array_t, parent: Any):
        """
        :param mts_array: raw array to wrap in a Python-compatible class
        :param parent: owner of the raw array, we will keep a reference to this
            python object
        """
        shape_ptr = ctypes.POINTER(c_uintptr_t)()
        shape_count = c_uintptr_t()
        status = mts_array.shape(mts_array.ptr, shape_ptr, shape_count)
        check_status(status)

        shape = []
        for i in range(shape_count.value):
            shape.append(shape_ptr[i])

        # Use as_dlpack to get data pointer and dtype
        dl_managed_ptr = ctypes.POINTER(DLManagedTensorVersioned)()
        device = DLDevice(device_type=DLDeviceType.kDLCPU, device_id=0)
        version = DLPackVersion(major=1, minor=0)
        status = mts_array.as_dlpack(
            mts_array.ptr,
            ctypes.byref(dl_managed_ptr),
            device,
            None,
            version,
        )
        check_status(status)

        array = np.from_dlpack(DLPackArray(dl_managed_ptr))
        obj = array.view(cls)

        # keep a reference to the parent object (if any) to prevent it from
        # being garbage-collected too early.
        obj._parent = parent
        # prevent the DLPack tensor from being freed while we hold a view
        obj._dl_managed_ptr = dl_managed_ptr

        return obj

    def __array_finalize__(self, obj):
        # keep the parent around when creating sub-views of this array
        self._parent = getattr(obj, "_parent", None)
        self._dl_managed_ptr = getattr(obj, "_dl_managed_ptr", None)

    def __array_wrap__(self, new, context=None, return_scalar=False):
        self_ptr = self.ctypes.data
        self_size = self.nbytes

        new_ptr = new.ctypes.data

        if self_ptr <= new_ptr <= self_ptr + self_size:
            # if the new array is a view inside memory owned by self, wrap it in
            # a ExternalCpuArray
            return super().__array_wrap__(new)
        else:
            # return the ndarray straight away
            return np.asarray(new)


class ExternalCudaArray:
    """
    .. deprecated:: 0.2.7

        This class is no longer used by metatensor, use ``array_backend="torch"`` to
        access data from non-Python origins as :py:class:`torch.Tensor`.

    Factory that wraps non-Python data on a CUDA device as a ``torch.Tensor`` via
    DLPack, keeping a reference to a parent Python object to prevent use-after-free.

    This is the CUDA counterpart to :py:class:`ExternalCpuArray`, intended for data that
    lives in CUDA memory. Requires PyTorch.

    For CUDA data (``device_type=2``), we go through CuPy when available for true
    zero-copy import, then convert to a ``torch.Tensor``. If CuPy is not installed we
    fall back to ``torch.from_dlpack`` directly.
    """

    def __new__(
        cls,
        mts_array: mts_array_t,
        parent: Any,
        *,
        device_type: int = 2,
        device_id: int = 0,
    ):
        """
        :param mts_array: raw array to wrap in a Python-compatible class
        :param parent: owner of the raw array, we will keep a reference to this
            python object
        :param device_type: DLPack device type (default: 2 = kDLCUDA)
        :param device_id: device index (default: 0)
        """
        try:
            import torch  # noqa: F811
        except ImportError as e:
            raise ImportError(
                "ExternalCudaArray requires PyTorch; "
                "install it with `pip install torch`"
            ) from e

        dl_managed_ptr = ctypes.POINTER(DLManagedTensorVersioned)()
        device = DLDevice(device_type=device_type, device_id=device_id)
        version = DLPackVersion(major=1, minor=0)
        status = mts_array.as_dlpack(
            mts_array.ptr,
            ctypes.byref(dl_managed_ptr),
            device,
            None,
            version,
        )
        check_status(status)

        dlpack_array = DLPackArray(dl_managed_ptr)

        # For CUDA data, prefer CuPy for true zero-copy import of external
        # GPU memory, then convert to a torch.Tensor (also zero-copy via
        # __cuda_array_interface__).
        tensor = None
        if device_type == 2:  # kDLCUDA
            try:
                import cupy

                cupy_array = cupy.from_dlpack(dlpack_array)
                tensor = torch.as_tensor(cupy_array)
            except ImportError:
                pass

        if tensor is None:
            tensor = torch.from_dlpack(dlpack_array)

        # keep a reference to the parent object to prevent it from being
        # garbage-collected while the tensor is alive
        tensor._parent = parent

        return tensor


def _call_with_growing_buffer(callback, initial=1024):
    bufflen = initial

    while True:
        buffer = ctypes.create_string_buffer(bufflen)
        try:
            callback(buffer, bufflen)
            break
        except MetatensorError as e:
            if e.status == mts_status_t.MTS_BUFFER_SIZE_ERROR:
                # grow the buffer and retry
                bufflen *= 2
            else:
                raise
    return buffer.value.decode("utf8")
