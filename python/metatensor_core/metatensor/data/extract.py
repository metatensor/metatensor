import ctypes
import enum
from typing import Any, NewType, Union

import numpy as np

from .._c_api import (
    DLDevice,
    DLManagedTensorVersioned,
    DLPackVersion,
    c_uintptr_t,
    mts_array_t,
    mts_data_origin_t,
)
from ..status import _check_status
from ..utils import _call_with_growing_buffer, _ptr_to_ndarray
from .array import (
    _KNOWN_ARRAY_WRAPPERS,
    _origin_numpy,
    _origin_pytorch,
    _register_origin,
)


class DLPackDtypeCode(enum.IntEnum):
    """DLPack data-type codes (from ``dlpack.h``)."""

    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLComplex = 5
    kDLBool = 6


# DLPack dtype -> numpy dtype mapping, used by ExternalCpuArray
_DLPACK_TO_NUMPY = {
    (DLPackDtypeCode.kDLFloat, 16): np.float16,
    (DLPackDtypeCode.kDLFloat, 32): np.float32,
    (DLPackDtypeCode.kDLFloat, 64): np.float64,
    (DLPackDtypeCode.kDLInt, 8): np.int8,
    (DLPackDtypeCode.kDLInt, 16): np.int16,
    (DLPackDtypeCode.kDLInt, 32): np.int32,
    (DLPackDtypeCode.kDLInt, 64): np.int64,
    (DLPackDtypeCode.kDLUInt, 8): np.uint8,
    (DLPackDtypeCode.kDLUInt, 16): np.uint16,
    (DLPackDtypeCode.kDLUInt, 32): np.uint32,
    (DLPackDtypeCode.kDLUInt, 64): np.uint64,
    (DLPackDtypeCode.kDLBool, 8): np.bool_,
    (DLPackDtypeCode.kDLComplex, 64): np.complex64,
    (DLPackDtypeCode.kDLComplex, 128): np.complex128,
}


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    # This NewType is only used for typechecking and documentation purposes. If you are
    # trying to add support for new array types, see `data.array.ArrayWrapper` instead.
    Array = NewType("Array", Union[np.ndarray, torch.Tensor])
else:
    Array = NewType("Array", np.ndarray)

Array.__doc__ = """
An ``Array`` contains the actual data stored in a :py:class:`metatensor.TensorBlock`.

This data is manipulated by ``metatensor`` in a completely opaque way: this library does
not know what's inside the arrays appart from a small set of constrains:

- the array contains numeric data (:py:func:`metatensor.load` and
  :py:func:`metatensor.save` additionally requires contiguous arrays of 64-bit IEEE-754
  floating points numbers);
- they are stored as row-major, n-dimensional arrays with at least 2 dimensions;
- it is possible to create new arrays and move data from one array to another.

The actual type of an ``Array`` depends on how the :py:class:`metatensor.TensorBlock`
was created. Currently, :py:class:`numpy.ndarray` and :py:class:`torch.Tensor` are
supported.
"""


_ADDITIONAL_ORIGINS = {}


def register_external_data_wrapper(origin, klass):
    """
    Register a non-Python data origin and the corresponding class wrapper.

    The wrapper class constructor must take two arguments (raw ``mts_array`` and python
    ``parent`` object) and return a subclass of either :py:class:`numpy.ndarray` or
    :py:class:`torch.Tensor`, which keeps ``parent`` alive. The
    :py:class:`metatensor.data.ExternalCpuArray` class should provide the right behavior
    for data living in CPU memory, and can serve as an example for more advanced custom
    arrays.

    :param origin: data origin name as a string, corresponding to the output of
        :c:func:`mts_array_t.origin`
    :param klass: wrapper class to use for this origin
    """

    if not isinstance(origin, str):
        raise ValueError(f"origin must be a string, got {type(origin)}")

    global _ADDITIONAL_ORIGINS
    _ADDITIONAL_ORIGINS[_register_origin(origin)] = klass


def mts_array_to_python_array(mts_array, parent=None):
    """Convert a raw mts_array to a Python ``Array``.

    Either the underlying array was allocated by Python, and the Python object
    is directly returned; or the underlying array was not allocated by Python,
    and additional origins are searched for a suitable Python wrapper class.
    """
    origin = data_origin(mts_array)
    if _is_python_origin(origin):
        return _KNOWN_ARRAY_WRAPPERS[mts_array.ptr].array
    elif origin in _ADDITIONAL_ORIGINS:
        return _ADDITIONAL_ORIGINS[origin](mts_array, parent=parent)
    else:
        raise ValueError(
            f"unable to handle data coming from '{data_origin_name(origin)}', "
            "you should maybe register a new array wrapper with metatensor"
        )


def mts_array_was_allocated_by_python(mts_array):
    """Check if a given mts_array was allocated by Python"""
    return _is_python_origin(data_origin(mts_array))


def _is_python_origin(origin):
    return origin in [_origin_numpy(), _origin_pytorch()]


def data_origin(mts_array):
    """Get the data origin of an mts_array"""
    origin = mts_data_origin_t()
    mts_array.origin(mts_array.ptr, origin)
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
        _check_status(status)

        shape = []
        for i in range(shape_count.value):
            shape.append(shape_ptr[i])

        # Use as_dlpack to get data pointer and dtype
        dl_managed_ptr = ctypes.POINTER(DLManagedTensorVersioned)()
        device = DLDevice(device_type=1, device_id=0)  # kDLCPU
        version = DLPackVersion(major=1, minor=0)
        status = mts_array.as_dlpack(
            mts_array.ptr,
            ctypes.byref(dl_managed_ptr),
            device,
            None,
            version,
        )
        _check_status(status)

        dl_tensor = dl_managed_ptr.contents.dl_tensor
        data_ptr = dl_tensor.data
        dtype_code = dl_tensor.dtype.code
        dtype_bits = dl_tensor.dtype.bits

        np_dtype = _DLPACK_TO_NUMPY.get((dtype_code, dtype_bits))
        if np_dtype is None:
            raise ValueError(
                f"unsupported DLPack dtype: code={dtype_code}, bits={dtype_bits}"
            )

        c_type = np.ctypeslib.as_ctypes_type(np.dtype(np_dtype))
        array = _ptr_to_ndarray(
            ctypes.cast(data_ptr, ctypes.POINTER(c_type)),
            shape,
            np_dtype,
        )
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
    Factory that wraps non-Python data on a CUDA device as a ``torch.Tensor``
    via DLPack, keeping a reference to a parent Python object to prevent
    use-after-free.

    This is the CUDA counterpart to :py:class:`ExternalCpuArray`, intended for
    data that lives in CUDA memory. Requires PyTorch.

    For CUDA data (``device_type=2``), we go through CuPy when available for
    true zero-copy import, then convert to a ``torch.Tensor``. If CuPy is not
    installed we fall back to ``torch.from_dlpack`` directly.
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
        _check_status(status)

        from ._dlpack import (
            make_dlpack_unversioned_capsule,
            make_dlpack_versioned_capsule,
            wrap_versioned_as_unversioned,
        )

        capsule = make_dlpack_versioned_capsule(dl_managed_ptr)

        # For CUDA data, prefer CuPy for true zero-copy import of external
        # GPU memory, then convert to a torch.Tensor (also zero-copy via
        # __cuda_array_interface__).
        tensor = None
        if device_type == 2:  # kDLCUDA
            try:
                import cupy

                cupy_array = cupy.from_dlpack(capsule)
                tensor = torch.as_tensor(cupy_array)
            except ImportError:
                pass

        if tensor is None:
            try:
                tensor = torch.from_dlpack(capsule)
            except RuntimeError:
                # Older PyTorch (< 2.4) doesn't understand versioned
                # DLPack capsules.  Fall back to an unversioned capsule.
                unversioned = wrap_versioned_as_unversioned(dl_managed_ptr)
                capsule = make_dlpack_unversioned_capsule(unversioned)
                tensor = torch.from_dlpack(capsule)

        # keep a reference to the parent object to prevent it from being
        # garbage-collected while the tensor is alive
        tensor._parent = parent

        return tensor
