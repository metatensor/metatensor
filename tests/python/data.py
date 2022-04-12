import unittest
import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import ctypes

from aml_storage import data
from aml_storage._c_api import c_uintptr_t, AML_SUCCESS, aml_array_t


class TestAmlDataMixin:
    def test_origin(self):
        array = self.create_array((2, 3, 4))
        aml_data = data.AmlData(array)
        aml_array = aml_data.aml_array

        self.assertEqual(
            id(data.aml_array_to_python_object(aml_array).array),
            id(array),
        )

        origin = data.data_origin(aml_array)
        self.assertEqual(data.data_origin_name(origin), self.expected_origin())

    def test_shape(self):
        array = self.create_array((2, 3, 4))
        aml_data = data.AmlData(array)
        aml_array = aml_data.aml_array

        self.assertEqual(_get_shape(aml_array, self), [2, 3, 4])

        new_shape = ctypes.ARRAY(c_uintptr_t, 4)(2, 3, 2, 2)
        status = aml_array.reshape(aml_array.ptr, new_shape, len(new_shape))
        self.assertEqual(status, AML_SUCCESS)

        self.assertEqual(_get_shape(aml_array, self), [2, 3, 2, 2])

    def test_swap_axes(self):
        array = self.create_array((2, 3, 18, 23))
        aml_data = data.AmlData(array)
        aml_array = aml_data.aml_array

        aml_array.swap_axes(aml_array.ptr, 1, 3)
        self.assertEqual(_get_shape(aml_array, self), [2, 23, 18, 3])

    def test_create(self):
        # TODO

        # include tests for destroy here
        pass

    def test_copy(self):
        array = self.create_array((2, 3, 4))
        array[1, :, :] = 3
        array[1, 2, :] = 5
        aml_data = data.AmlData(array)
        aml_array = aml_data.aml_array

        copy = aml_array_t()
        status = aml_array.copy(aml_array.ptr, copy)
        self.assertEqual(status, AML_SUCCESS)

        array_copy = data.aml_array_to_python_object(copy).array
        self.assertNotEqual(id(array_copy), id(array))

        self.assertTrue(np.all(np.array(array_copy) == np.array(array)))

    def test_move_sample(self):
        array = self.create_array((2, 3, 8))
        array[:] = 4.0
        aml_data = data.AmlData(array)
        aml_array = aml_data.aml_array

        other = self.create_array((1, 3, 4))
        other[:] = 2.0
        aml_data_other = data.AmlData(other)
        aml_array_other = aml_data_other.aml_array

        aml_array.move_sample(aml_array.ptr, 1, 3, 7, aml_array_other.ptr, 0)
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


class TestNumpyData(unittest.TestCase, TestAmlDataMixin):
    def expected_origin(self):
        return "aml_storage.data.numpy"

    def create_array(self, shape):
        return np.zeros(shape)


if HAS_TORCH:

    class TestTorchData(unittest.TestCase, TestAmlDataMixin):
        def expected_origin(self):
            return "aml_storage.data.torch"

        def create_array(self, shape):
            return torch.zeros(shape, device="cpu")


def _get_shape(aml_array, test):
    shape_ptr = ctypes.POINTER(c_uintptr_t)()
    shape_count = c_uintptr_t()
    status = aml_array.shape(aml_array.ptr, shape_ptr, shape_count)

    test.assertEqual(status, AML_SUCCESS)

    shape = []
    for i in range(shape_count.value):
        shape.append(shape_ptr[i])

    return shape


if __name__ == "__main__":
    unittest.main()
