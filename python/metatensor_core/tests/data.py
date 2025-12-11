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
        assert managed.version.minor == 0
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
