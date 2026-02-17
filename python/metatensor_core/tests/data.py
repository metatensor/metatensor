import ctypes
import gc
import os
import weakref

import numpy as np
import pytest
from numpy.testing import assert_equal


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import metatensor
from metatensor._c_api import (
    MTS_SUCCESS,
    DLDevice,
    DLManagedTensorVersioned,
    DLPackVersion,
    c_uintptr_t,
    mts_array_t,
    mts_sample_mapping_t,
)


# Kanged from the metatensor-torch _test_utils
def can_use_mps_backend():
    return (
        # Github Actions M1 runners don't have a GPU accessible
        os.environ.get("GITHUB_ACTIONS") is None
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def free_mts_array(array):
    array.destroy(array.ptr)


class MtsArrayMixin:
    def test_origin(self):
        array = self.create_array((2, 3, 4))
        mts_array = metatensor.data.create_mts_array(array)

        assert id(metatensor.data.mts_array_to_python_array(mts_array)) == id(array)

        origin = metatensor.data.data_origin(mts_array)
        assert metatensor.data.data_origin_name(origin) == self.expected_origin()

        free_mts_array(mts_array)

    def test_shape(self):
        array = self.create_array((2, 3, 4))
        mts_array = metatensor.data.create_mts_array(array)

        assert _get_shape(mts_array, self) == [2, 3, 4]

        new_shape = ctypes.ARRAY(c_uintptr_t, 4)(2, 3, 2, 2)
        status = mts_array.reshape(mts_array.ptr, new_shape, len(new_shape))
        assert status == MTS_SUCCESS

        assert _get_shape(mts_array, self) == [2, 3, 2, 2]

        free_mts_array(mts_array)

    def test_swap_axes(self):
        array = self.create_array((2, 3, 18, 23))
        mts_array = metatensor.data.create_mts_array(array)

        mts_array.swap_axes(mts_array.ptr, 1, 3)
        assert _get_shape(mts_array, self) == [2, 23, 18, 3]

        free_mts_array(mts_array)

    def test_create(self):
        array = self.create_array((2, 3))
        array_ref = weakref.ref(array)

        mts_array = metatensor.data.create_mts_array(array)

        new_mts_array = mts_array_t()
        new_shape = ctypes.ARRAY(c_uintptr_t, 2)(18, 4)
        status = mts_array.create(
            mts_array.ptr, new_shape, len(new_shape), new_mts_array
        )
        assert status == MTS_SUCCESS

        new_array = metatensor.data.mts_array_to_python_array(new_mts_array)
        assert id(new_array) != id(array)

        assert_equal(new_array, np.zeros((18, 4)))

        del array
        gc.collect()

        # there is still one reference to the array through mts_array
        assert array_ref() is not None

        free_mts_array(mts_array)
        del mts_array
        gc.collect()

        assert array_ref() is None

        free_mts_array(new_mts_array)

    def test_copy(self):
        array = self.create_array((2, 3, 4))
        array[1, :, :] = 3
        array[1, 2, :] = 5
        mts_array = metatensor.data.create_mts_array(array)

        copy = mts_array_t()
        status = mts_array.copy(mts_array.ptr, copy)
        assert status == MTS_SUCCESS

        array_copy = metatensor.data.mts_array_to_python_array(copy)
        assert id(array_copy) != id(array)

        assert_equal(self.to_numpy(array_copy), self.to_numpy(array))

        free_mts_array(mts_array)
        free_mts_array(copy)

    def test_move_samples_from(self):
        array = self.create_array((2, 3, 8))
        array[:] = 4.0
        mts_array = metatensor.data.create_mts_array(array)

        other = self.create_array((1, 3, 4))
        other[:] = 2.0
        mts_array_other = metatensor.data.create_mts_array(other)

        move = mts_sample_mapping_t(input=0, output=1)
        move_array = ctypes.ARRAY(mts_sample_mapping_t, 1)(move)

        mts_array.move_samples_from(
            mts_array.ptr,
            mts_array_other.ptr,
            move_array,
            len(move_array),
            3,
            7,
        )
        expected = np.array(
            [
                # unmodified first sample
                [
                    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                ],
                # second sample changed for properties 3:7
                [
                    [4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 4.0],
                    [4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 4.0],
                    [4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 4.0],
                ],
            ]
        )
        assert_equal(array, expected)

        free_mts_array(mts_array)
        free_mts_array(mts_array_other)

    def get_available_devices():
        devices = [("cpu", DLDevice(1, 0))]
        if HAS_TORCH and torch.cuda.is_available():
            devices.append(("cuda", DLDevice(2, 0)))
        if can_use_mps_backend():
            devices.append(("mps", DLDevice(8, 0)))
        return devices

    @pytest.mark.parametrize("device, dldevice", get_available_devices())
    def test_dlpack(self, device, dldevice):
        # Create a sample array
        array = self.create_array((2, 3))
        if self.expected_origin() == "metatensor.data.array.numpy" and device != "cpu":
            pytest.skip("NumPy only supports CPU")
        if not device:
            pytest.skip()
        if isinstance(array, torch.Tensor):
            array = torch.asarray(array, device=device)
        mts_array = metatensor.data.create_mts_array(array)

        # Prepare a pointer to receive the DLManagedTensorVersioned
        dl_managed_ptr = ctypes.POINTER(DLManagedTensorVersioned)()
        version = DLPackVersion(1, 0)

        status = mts_array.as_dlpack(
            mts_array.ptr,
            ctypes.byref(dl_managed_ptr),
            dldevice,
            None,  # stream
            version,
        )

        # Check we got a valid pointer back
        assert status == MTS_SUCCESS
        assert bool(dl_managed_ptr)

        # Dereference to check contents
        managed = dl_managed_ptr.contents
        dl_tensor = managed.dl_tensor

        # Verify Metadata
        assert managed.version.major == 1
        assert managed.version.minor <= 3
        assert dl_tensor.ndim == 2
        assert dl_tensor.shape[0] == 2
        assert dl_tensor.shape[1] == 3

        # Verify Data Pointer matches the source array
        if self.expected_origin() == "metatensor.data.array.numpy":
            assert dl_tensor.data == array.ctypes.data
        elif self.expected_origin() == "metatensor.data.array.torch":
            assert dl_tensor.data == array.data_ptr()

        # IMPORTANT: Call the deleter to cleanup the C-side resources (and refcounts)
        # This emulates what a consumer (like another library) would do.
        managed.deleter(dl_managed_ptr)

        free_mts_array(mts_array)


class TestNumpyData(MtsArrayMixin):
    def expected_origin(self):
        return "metatensor.data.array.numpy"

    def create_array(self, shape):
        return np.zeros(shape)

    def to_numpy(self, array):
        return np.array(array)


if HAS_TORCH:

    class TestTorchData(MtsArrayMixin):
        def expected_origin(self):
            return "metatensor.data.array.torch"

        def create_array(self, shape):
            return torch.zeros(shape, device="cpu")

        def to_numpy(self, array):
            return array.numpy()


def _get_shape(mts_array, test):
    shape_ptr = ctypes.POINTER(c_uintptr_t)()
    shape_count = c_uintptr_t()
    status = mts_array.shape(mts_array.ptr, shape_ptr, shape_count)

    assert status == MTS_SUCCESS

    shape = []
    for i in range(shape_count.value):
        shape.append(shape_ptr[i])

    return shape


def test_dlpack_dtype_code_enum():
    """Verify that the DLPackDtypeCode enum has the correct values from dlpack.h."""
    from metatensor.data.extract import DLPackDtypeCode

    assert DLPackDtypeCode.kDLInt == 0
    assert DLPackDtypeCode.kDLUInt == 1
    assert DLPackDtypeCode.kDLFloat == 2
    assert DLPackDtypeCode.kDLComplex == 5
    assert DLPackDtypeCode.kDLBool == 6

    # Verify they work as dict keys (used in _DLPACK_TO_NUMPY)
    d = {(DLPackDtypeCode.kDLFloat, 64): "f64"}
    assert d[(2, 64)] == "f64"


def test_external_cuda_array_importable():
    """ExternalCudaArray should be importable from the public API."""
    from metatensor.data import ExternalCudaArray

    assert ExternalCudaArray is not None


def test_external_cuda_array_requires_torch(monkeypatch):
    """ExternalCudaArray.__new__ should raise ImportError when torch is missing."""
    import builtins

    from metatensor.data import extract

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    dummy_mts_array = mts_array_t()  # won't be used — error raised first
    with pytest.raises(ImportError, match="ExternalCudaArray requires PyTorch"):
        extract.ExternalCudaArray(dummy_mts_array, parent=None)


@pytest.mark.skipif(not HAS_TORCH, reason="requires PyTorch")
def test_external_cuda_array_cpu_path():
    """ExternalCudaArray should work with device_type=1 (CPU) for testing."""
    from metatensor.data.extract import ExternalCudaArray

    # Create a numpy array and wrap it as mts_array
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    mts_array = metatensor.data.create_mts_array(arr)

    # Use device_type=1 (kDLCPU) to test the full code path without a GPU
    result = ExternalCudaArray(mts_array, parent=None, device_type=1, device_id=0)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 2)
    assert result.device.type == "cpu"
    np.testing.assert_array_equal(result.numpy(), arr)

    free_mts_array(mts_array)


def test_dlpack_to_numpy_mapping():
    """Verify _DLPACK_TO_NUMPY covers common dtypes and matches numpy equivalents."""
    from metatensor.data.extract import _DLPACK_TO_NUMPY, DLPackDtypeCode

    # float types
    assert _DLPACK_TO_NUMPY[(DLPackDtypeCode.kDLFloat, 32)] is np.float32
    assert _DLPACK_TO_NUMPY[(DLPackDtypeCode.kDLFloat, 64)] is np.float64

    # integer types
    assert _DLPACK_TO_NUMPY[(DLPackDtypeCode.kDLInt, 32)] is np.int32
    assert _DLPACK_TO_NUMPY[(DLPackDtypeCode.kDLInt, 64)] is np.int64

    # unsigned int
    assert _DLPACK_TO_NUMPY[(DLPackDtypeCode.kDLUInt, 8)] is np.uint8

    # bool
    assert _DLPACK_TO_NUMPY[(DLPackDtypeCode.kDLBool, 8)] is np.bool_

    # complex
    assert _DLPACK_TO_NUMPY[(DLPackDtypeCode.kDLComplex, 128)] is np.complex128

    # unsupported dtype returns None
    assert _DLPACK_TO_NUMPY.get((99, 99)) is None


def test_make_dlpack_versioned_capsule():
    """Test PyCapsule creation from DLManagedTensorVersioned pointer."""
    from metatensor.data._dlpack import (
        DLPACK_VERSIONED_NAME,
        PYTHON_API,
        make_dlpack_versioned_capsule,
    )

    # Create a numpy array and get a DLPack tensor from it
    arr = np.zeros((2, 3), dtype=np.float64)
    mts_array = metatensor.data.create_mts_array(arr)

    dl_managed_ptr = ctypes.POINTER(DLManagedTensorVersioned)()
    device = DLDevice(1, 0)  # kDLCPU
    version = DLPackVersion(1, 0)
    status = mts_array.as_dlpack(
        mts_array.ptr,
        ctypes.byref(dl_managed_ptr),
        device,
        None,
        version,
    )
    assert status == MTS_SUCCESS

    capsule = make_dlpack_versioned_capsule(dl_managed_ptr)
    assert capsule is not None

    # Verify capsule name
    name = PYTHON_API.PyCapsule_GetName(capsule)
    assert name == DLPACK_VERSIONED_NAME


def test_make_dlpack_versioned_capsule_null():
    """make_dlpack_versioned_capsule should raise on null pointer."""
    from metatensor.data._dlpack import make_dlpack_versioned_capsule

    null_ptr = ctypes.POINTER(DLManagedTensorVersioned)()
    with pytest.raises(ValueError, match="null"):
        make_dlpack_versioned_capsule(null_ptr)


def test_mmap_cpu_array_type():
    """MmapCpuArray values from load_mmap should be the right type."""
    from metatensor.data.extract import MmapCpuArray

    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "data.mts",
    )
    tensor = metatensor.load_mmap(path)
    values = tensor.block(0).values

    assert isinstance(values, MmapCpuArray)
    assert isinstance(values, np.ndarray)
    assert values._parent is not None
    assert values._dl_managed_ptr is not None


def test_mmap_cpu_array_subview():
    """Subviews of MmapCpuArray should keep parent alive."""
    from metatensor.data.extract import MmapCpuArray

    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "data.mts",
    )
    tensor = metatensor.load_mmap(path)
    values = tensor.block(0).values

    # Slice should still be MmapCpuArray with parent
    subview = values[::2]
    assert isinstance(subview, MmapCpuArray)
    assert subview._parent is not None

    # Operations that produce new memory should return plain ndarray
    transformed = np.sum(values, axis=0)
    assert not isinstance(transformed, MmapCpuArray)
    assert isinstance(transformed, np.ndarray)


def test_mmap_cpu_array_keepalive():
    """MmapCpuArray must keep its parent alive to prevent use-after-free."""
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "block.mts",
    )

    block = metatensor.load_block_mmap(path)
    values = block.values

    block_ref = weakref.ref(block)
    del block
    gc.collect()

    # values should keep the block alive
    assert block_ref() is not None
    assert values.shape == (9, 5, 3)

    # subview should still keep things alive
    sub = values[0:2]
    del values
    gc.collect()
    assert block_ref() is not None
    assert sub.shape[0] == 2


def test_ensure_mmap_origin_registered_idempotent():
    """Calling _ensure_mmap_origin_registered twice should not fail."""
    from metatensor.data.extract import _ensure_mmap_origin_registered

    # First call registers
    _ensure_mmap_origin_registered()
    # Second call is a no-op
    _ensure_mmap_origin_registered()


def test_array_from_dl_tensor_unsupported_dtype():
    """_array_from_dl_tensor should raise on unsupported dtype."""
    from metatensor._c_api import DLDataType, DLTensor
    from metatensor.data.extract import _array_from_dl_tensor

    # Create a fake DLTensor with an unsupported dtype
    fake_tensor = DLTensor()
    fake_tensor.dtype = DLDataType(code=99, bits=99, lanes=1)
    fake_tensor.ndim = 0

    with pytest.raises(ValueError, match="unsupported DLPack dtype"):
        _array_from_dl_tensor(fake_tensor)


def test_array_from_dl_tensor_happy_path():
    """_array_from_dl_tensor should correctly create a numpy array from a DLTensor."""
    from metatensor.data.extract import _array_from_dl_tensor

    # Create a real array and get a DLTensor via as_dlpack
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    mts_array = metatensor.data.create_mts_array(arr)

    dl_managed_ptr = ctypes.POINTER(DLManagedTensorVersioned)()
    device = DLDevice(1, 0)
    version = DLPackVersion(1, 0)
    status = mts_array.as_dlpack(
        mts_array.ptr,
        ctypes.byref(dl_managed_ptr),
        device,
        None,
        version,
    )
    assert status == MTS_SUCCESS

    result = _array_from_dl_tensor(dl_managed_ptr.contents.dl_tensor)
    assert result.shape == (2, 3)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, arr)

    # Clean up
    managed = dl_managed_ptr.contents
    managed.deleter(dl_managed_ptr)
    free_mts_array(mts_array)


def test_wrap_unversioned_as_versioned():
    """wrap_unversioned_as_versioned should produce a valid versioned DLPack struct."""
    from metatensor._c_api import DLDataType
    from metatensor.data._dlpack import (
        DLManagedTensor,
        _DLManagedTensorDeleter,
        wrap_unversioned_as_versioned,
    )

    # Build a minimal DLManagedTensor with a real data pointer
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    data_ptr = arr.ctypes.data

    unversioned = DLManagedTensor()
    unversioned.dl_tensor.data = data_ptr
    unversioned.dl_tensor.ndim = 2
    shape = (ctypes.c_int64 * 2)(2, 2)
    unversioned.dl_tensor.shape = ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64))
    unversioned.dl_tensor.strides = None
    unversioned.dl_tensor.byte_offset = 0
    unversioned.dl_tensor.dtype = DLDataType(code=2, bits=64, lanes=1)  # kDLFloat
    unversioned.dl_tensor.device.device_type = 1  # kDLCPU
    unversioned.dl_tensor.device.device_id = 0

    # Set a no-op deleter
    @_DLManagedTensorDeleter
    def noop_deleter(ptr):
        pass

    unversioned.deleter = noop_deleter
    unversioned.manager_ctx = None

    unversioned_ptr = ctypes.pointer(unversioned)
    versioned_ptr = wrap_unversioned_as_versioned(unversioned_ptr)

    versioned = versioned_ptr.contents
    assert versioned.version.major == 1
    assert versioned.version.minor == 0
    assert versioned.dl_tensor.ndim == 2
    assert versioned.dl_tensor.data == data_ptr
    assert versioned.dl_tensor.dtype.code == 2
    assert versioned.dl_tensor.dtype.bits == 64

    # Clean up: call the versioned deleter
    versioned.deleter(versioned_ptr)


def test_mmap_cpu_array_numpy_lt2_fallback(monkeypatch):
    """Test the NumPy < 2.0 fallback path in MmapCpuArray."""
    from metatensor.data.extract import MmapCpuArray

    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "data.mts",
    )

    # Make np.from_dlpack raise ValueError to simulate NumPy < 2.0
    def mock_from_dlpack(obj):
        raise ValueError("simulated NumPy < 2.0 failure")

    monkeypatch.setattr(np, "from_dlpack", mock_from_dlpack)

    tensor = metatensor.load_mmap(path)
    values = tensor.block(0).values

    assert isinstance(values, MmapCpuArray)
    assert isinstance(values, np.ndarray)
    assert values.shape[0] > 0
    assert values.dtype == np.float64


TEST_ORIGIN = metatensor.data.array._register_origin("python.test-origin")
metatensor.data.register_external_data_wrapper(
    "python.test-origin",
    metatensor.data.extract.ExternalCpuArray,
)


@metatensor.utils.catch_exceptions
def get_test_origin(this, origin):
    origin[0] = TEST_ORIGIN


GET_TEST_ORIGIN = metatensor.data.array._cast_to_ctype_functype(
    get_test_origin, "origin"
)


@metatensor.utils.catch_exceptions
def create_test_array(shape_ptr, shape_count, array):
    shape = []
    for i in range(shape_count):
        shape.append(shape_ptr[i])

    data = np.zeros(shape)
    mts_array = metatensor.data.create_mts_array(data)
    mts_array.origin = GET_TEST_ORIGIN

    array[0] = mts_array


def test_parent_keepalive():
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "data.mts",
    )
    tensor = metatensor.io.load_custom_array(path.encode("utf8"), create_test_array)

    values = tensor.block(0).values
    assert isinstance(values, metatensor.data.extract.ExternalCpuArray)
    assert values._parent is not None

    tensor_ref = weakref.ref(tensor)
    del tensor
    gc.collect()

    # values should keep the tensor alive
    assert tensor_ref() is not None

    view = values[::2]
    del values
    gc.collect()

    # view should keep the tensor alive
    assert tensor_ref() is not None

    transformed = np.sum(view)
    del view
    gc.collect()

    # transformed should NOT keep the tensor alive
    assert tensor_ref() is None

    assert np.isclose(transformed, 1.1596965632269784)
