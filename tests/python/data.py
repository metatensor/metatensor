import unittest
import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import ctypes

from equistore import data
from equistore._c_api import c_uintptr_t, EQS_SUCCESS, eqs_array_t, eqs_sample_move_t


class TestArrayWrapperMixin:
    def test_origin(self):
        array = self.create_array((2, 3, 4))
        wrapper = data.ArrayWrapper(array)
        eqs_array = wrapper.eqs_array

        self.assertEqual(
            id(data.eqs_array_to_python_object(eqs_array).array),
            id(array),
        )

        origin = data.data_origin(eqs_array)
        self.assertEqual(data.data_origin_name(origin), self.expected_origin())

    def test_shape(self):
        array = self.create_array((2, 3, 4))
        wrapper = data.ArrayWrapper(array)
        eqs_array = wrapper.eqs_array

        self.assertEqual(_get_shape(eqs_array, self), [2, 3, 4])

        new_shape = ctypes.ARRAY(c_uintptr_t, 4)(2, 3, 2, 2)
        status = eqs_array.reshape(eqs_array.ptr, new_shape, len(new_shape))
        self.assertEqual(status, EQS_SUCCESS)

        self.assertEqual(_get_shape(eqs_array, self), [2, 3, 2, 2])

    def test_swap_axes(self):
        array = self.create_array((2, 3, 18, 23))
        wrapper = data.ArrayWrapper(array)
        eqs_array = wrapper.eqs_array

        eqs_array.swap_axes(eqs_array.ptr, 1, 3)
        self.assertEqual(_get_shape(eqs_array, self), [2, 23, 18, 3])

    def test_create(self):
        # TODO

        # include tests for destroy here
        pass

    def test_copy(self):
        array = self.create_array((2, 3, 4))
        array[1, :, :] = 3
        array[1, 2, :] = 5
        wrapper = data.ArrayWrapper(array)
        eqs_array = wrapper.eqs_array

        copy = eqs_array_t()
        status = eqs_array.copy(eqs_array.ptr, copy)
        self.assertEqual(status, EQS_SUCCESS)

        array_copy = data.eqs_array_to_python_object(copy).array
        self.assertNotEqual(id(array_copy), id(array))

        self.assertTrue(np.all(np.array(array_copy) == np.array(array)))

    def test_move_samples_from(self):
        array = self.create_array((2, 3, 8))
        array[:] = 4.0
        wrapper = data.ArrayWrapper(array)
        eqs_array = wrapper.eqs_array

        other = self.create_array((1, 3, 4))
        other[:] = 2.0
        wrapper_other = data.ArrayWrapper(other)
        eqs_array_other = wrapper_other.eqs_array

        move = eqs_sample_move_t(input=0, output=1)
        move_array = ctypes.ARRAY(eqs_sample_move_t, 1)(move)

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
        self.assertTrue(np.all(np.array(array) == expected))


class TestNumpyData(unittest.TestCase, TestArrayWrapperMixin):
    def expected_origin(self):
        return "equistore.data.numpy"

    def create_array(self, shape):
        return np.zeros(shape)


if HAS_TORCH:

    class TestTorchData(unittest.TestCase, TestArrayWrapperMixin):
        def expected_origin(self):
            return "equistore.data.torch"

        def create_array(self, shape):
            return torch.zeros(shape, device="cpu")


def _get_shape(eqs_array, test):
    shape_ptr = ctypes.POINTER(c_uintptr_t)()
    shape_count = c_uintptr_t()
    status = eqs_array.shape(eqs_array.ptr, shape_ptr, shape_count)

    test.assertEqual(status, EQS_SUCCESS)

    shape = []
    for i in range(shape_count.value):
        shape.append(shape_ptr[i])

    return shape


if __name__ == "__main__":
    unittest.main()
