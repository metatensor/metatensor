import gc
import os
import weakref

import numpy as np
from numpy.testing import assert_equal


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import ctypes

import equistore.core
from equistore.core._c_api import (
    EQS_SUCCESS,
    c_uintptr_t,
    eqs_array_t,
    eqs_sample_mapping_t,
)


def free_eqs_array(array):
    array.destroy(array.ptr)


class ArrayWrapperMixin:
    def test_origin(self):
        array = self.create_array((2, 3, 4))
        wrapper = equistore.core.data.ArrayWrapper(array)
        eqs_array = wrapper.into_eqs_array()

        assert id(equistore.core.data.eqs_array_to_python_array(eqs_array)) == id(array)

        origin = equistore.core.data.data_origin(eqs_array)
        assert equistore.core.data.data_origin_name(origin) == self.expected_origin()

        free_eqs_array(eqs_array)

    def test_shape(self):
        array = self.create_array((2, 3, 4))
        wrapper = equistore.core.data.ArrayWrapper(array)
        eqs_array = wrapper.into_eqs_array()

        assert _get_shape(eqs_array, self) == [2, 3, 4]

        new_shape = ctypes.ARRAY(c_uintptr_t, 4)(2, 3, 2, 2)
        status = eqs_array.reshape(eqs_array.ptr, new_shape, len(new_shape))
        assert status == EQS_SUCCESS

        assert _get_shape(eqs_array, self) == [2, 3, 2, 2]

        free_eqs_array(eqs_array)

    def test_swap_axes(self):
        array = self.create_array((2, 3, 18, 23))
        wrapper = equistore.core.data.ArrayWrapper(array)
        eqs_array = wrapper.into_eqs_array()

        eqs_array.swap_axes(eqs_array.ptr, 1, 3)
        assert _get_shape(eqs_array, self) == [2, 23, 18, 3]

        free_eqs_array(eqs_array)

    def test_create(self):
        array = self.create_array((2, 3))
        array_ref = weakref.ref(array)

        wrapper = equistore.core.data.ArrayWrapper(array)
        eqs_array = wrapper.into_eqs_array()

        new_eqs_array = eqs_array_t()
        new_shape = ctypes.ARRAY(c_uintptr_t, 2)(18, 4)
        status = eqs_array.create(
            eqs_array.ptr, new_shape, len(new_shape), new_eqs_array
        )
        assert status == EQS_SUCCESS

        new_array = equistore.core.data.eqs_array_to_python_array(new_eqs_array)
        assert id(new_array) != id(array)

        assert_equal(new_array, np.zeros((18, 4)))

        del array
        del wrapper
        gc.collect()

        # there is still one reference to the array through eqs_array
        assert array_ref() is not None

        free_eqs_array(eqs_array)
        del eqs_array
        gc.collect()

        assert array_ref() is None

        free_eqs_array(new_eqs_array)

    def test_copy(self):
        array = self.create_array((2, 3, 4))
        array[1, :, :] = 3
        array[1, 2, :] = 5
        wrapper = equistore.core.data.ArrayWrapper(array)
        eqs_array = wrapper.into_eqs_array()

        copy = eqs_array_t()
        status = eqs_array.copy(eqs_array.ptr, copy)
        assert status == EQS_SUCCESS

        array_copy = equistore.core.data.eqs_array_to_python_array(copy)
        assert id(array_copy) != id(array)

        assert_equal(np.array(array_copy), np.array(array))

        free_eqs_array(eqs_array)
        free_eqs_array(copy)

    def test_move_samples_from(self):
        array = self.create_array((2, 3, 8))
        array[:] = 4.0
        wrapper = equistore.core.data.ArrayWrapper(array)
        eqs_array = wrapper.into_eqs_array()

        other = self.create_array((1, 3, 4))
        other[:] = 2.0
        wrapper_other = equistore.core.data.ArrayWrapper(other)
        eqs_array_other = wrapper_other.into_eqs_array()

        move = eqs_sample_mapping_t(input=0, output=1)
        move_array = ctypes.ARRAY(eqs_sample_mapping_t, 1)(move)

        eqs_array.move_samples_from(
            eqs_array.ptr,
            eqs_array_other.ptr,
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

        free_eqs_array(eqs_array)
        free_eqs_array(eqs_array_other)


class TestNumpyData(ArrayWrapperMixin):
    def expected_origin(self):
        return "equistore.core.data.array.numpy"

    def create_array(self, shape):
        return np.zeros(shape)


if HAS_TORCH:

    class TestTorchData(ArrayWrapperMixin):
        def expected_origin(self):
            return "equistore.core.data.array.torch"

        def create_array(self, shape):
            return torch.zeros(shape, device="cpu")


def _get_shape(eqs_array, test):
    shape_ptr = ctypes.POINTER(c_uintptr_t)()
    shape_count = c_uintptr_t()
    status = eqs_array.shape(eqs_array.ptr, shape_ptr, shape_count)

    assert status == EQS_SUCCESS

    shape = []
    for i in range(shape_count.value):
        shape.append(shape_ptr[i])

    return shape


TEST_ORIGIN = equistore.core.data.array._register_origin("python.test-origin")
equistore.core.data.register_external_data_wrapper(
    "python.test-origin",
    equistore.core.data.extract.ExternalCpuArray,
)


@equistore.core.utils.catch_exceptions
def create_test_array(shape_ptr, shape_count, array):
    shape = []
    for i in range(shape_count):
        shape.append(shape_ptr[i])

    data = np.zeros(shape)
    wrapper = equistore.core.data.array.ArrayWrapper(data)
    eqs_array = wrapper.into_eqs_array()

    @equistore.core.utils.catch_exceptions
    def test_origin(this, origin):
        origin[0] = TEST_ORIGIN

    eqs_array.origin = eqs_array.origin.__class__(test_origin)

    array[0] = eqs_array


def test_parent_keepalive():
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "equistore", "tests", "data.npz"
    )
    tensor = equistore.core.io.load_custom_array(path.encode("utf8"), create_test_array)

    values = tensor.block(0).values
    assert isinstance(values, equistore.core.data.extract.ExternalCpuArray)
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
