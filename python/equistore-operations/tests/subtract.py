import os
import unittest

import numpy as np

import equistore
from equistore import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestSubtract(unittest.TestCase):
    def test_self_subtract_tensors_nogradient(self):
        block_1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        block_2 = TensorBlock(
            values=np.array([[1, 2], [3, 4], [5, 6]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        block_3 = TensorBlock(
            values=np.array([[1.5, 2.1], [6.7, 10.2]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        block_4 = TensorBlock(
            values=np.array([[10, 200.8], [3.76, 4.432], [545, 26]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("p", 2),
        )

        block_res_1 = TensorBlock(
            values=np.array([[-0.5, -0.1], [-3.7, -5.2]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        block_res_2 = TensorBlock(
            values=np.array([[-9, -198.8], [-0.76, -0.432], [-540, -20]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
        A = TensorMap(keys, [block_1, block_2])
        B = TensorMap(keys, [block_3, block_4])
        A_copy = A.copy()
        B_copy = B.copy()
        tensor_sum = equistore.subtract(A, B)
        tensor_result = TensorMap(keys, [block_res_1, block_res_2])

        self.assertTrue(equistore.allclose(tensor_result, tensor_sum))
        # Check that the original tensors are not modified
        self.assertTrue(equistore.equal(A, A_copy))
        self.assertTrue(equistore.equal(B, B_copy))

    def test_self_subtract_tensors_gradient(self):
        block_1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        block_1.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array([[[6, 1], [7, 2]], [[8, 3], [9, 4]]]),
                samples=Labels.arange("sample", 2),
                components=[Labels.arange("c", 2)],
                properties=block_1.properties,
            ),
        )

        block_2 = TensorBlock(
            values=np.array([[-1, -2], [-3, -4], [5, 6]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        block_2.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array(
                    [
                        [[-10, -11], [-12, -13]],
                        [[14, 15], [10, 11]],
                        [[12, 13], [14, 15]],
                    ]
                ),
                samples=Labels.arange("sample", 3),
                components=[Labels.arange("c", 2)],
                properties=block_2.properties,
            ),
        )

        block_3 = TensorBlock(
            values=np.array([[1.5, 2.1], [6.7, 10.2]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        block_3.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array([[[-1, -0.1], [-2, -0.2]], [[-3, -0.3], [-4.5, -0.4]]]),
                samples=Labels.arange("sample", 2),
                components=[Labels.arange("c", 2)],
                properties=block_3.properties,
            ),
        )

        block_4 = TensorBlock(
            values=np.array([[10, 200.8], [3.76, 4.432], [-545, -26]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        block_4.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array(
                    [
                        [[1.0, 1.1], [1.2, 1.3]],
                        [[-1.4, -1.5], [-1.0, -1.1]],
                        [[1.2, 1.3], [1.4, 1.5]],
                    ]
                ),
                samples=Labels.arange("sample", 3),
                components=[Labels.arange("c", 2)],
                properties=block_4.properties,
            ),
        )

        block_res_1 = TensorBlock(
            values=np.array([[-0.5, -0.1], [-3.7, -5.2]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        block_res_1.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array(
                    np.array([[[7, 1.1], [9, 2.2]], [[11, 3.3], [13.5, 4.4]]])
                ),
                samples=Labels.arange("sample", 2),
                components=[Labels.arange("c", 2)],
                properties=block_res_1.properties,
            ),
        )

        block_res_2 = TensorBlock(
            values=np.array([[-11, -202.8], [-6.76, -8.432], [550, 32]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        block_res_2.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array(
                    [
                        [[-11.0, -12.1], [-13.2, -14.3]],
                        [[15.4, 16.5], [11.0, 12.1]],
                        [[10.8, 11.7], [12.6, 13.5]],
                    ]
                ),
                samples=Labels.arange("sample", 3),
                components=[Labels.arange("c", 2)],
                properties=block_res_2.properties,
            ),
        )

        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
        A = TensorMap(keys, [block_1, block_2])
        B = TensorMap(keys, [block_3, block_4])
        A_copy = A.copy()
        B_copy = B.copy()
        tensor_sum = equistore.subtract(A, B)
        tensor_result = TensorMap(keys, [block_res_1, block_res_2])

        self.assertTrue(equistore.allclose(tensor_result, tensor_sum))
        # Check that the original tensors are not modified
        self.assertTrue(equistore.equal(A, A_copy))
        self.assertTrue(equistore.equal(B, B_copy))

    def test_self_subtract_scalar_gradient(self):
        block_1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        block_1.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array([[[6, 1], [7, 2]], [[8, 3], [9, 4]]]),
                samples=Labels.arange("sample", 2),
                components=[Labels.arange("c", 2)],
                properties=block_1.properties,
            ),
        )

        block_2 = TensorBlock(
            values=np.array([[11, 12], [13, 14], [15, 16]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        block_2.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array(
                    [[[10, 11], [12, 13]], [[14, 15], [10, 11]], [[12, 13], [14, 15]]]
                ),
                samples=Labels.arange("sample", 3),
                components=[Labels.arange("c", 2)],
                properties=block_2.properties,
            ),
        )

        block_res_1 = TensorBlock(
            values=np.array([[6.1, 7.1], [8.1, 10.1]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        block_res_1.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array([[[6, 1], [7, 2]], [[8, 3], [9, 4]]]),
                samples=Labels.arange("sample", 2),
                components=[Labels.arange("c", 2)],
                properties=block_res_1.properties,
            ),
        )

        block_res_2 = TensorBlock(
            values=np.array([[16.1, 17.1], [18.1, 19.1], [20.1, 21.1]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        block_res_2.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array(
                    [[[10, 11], [12, 13]], [[14, 15], [10, 11]], [[12, 13], [14, 15]]]
                ),
                samples=Labels.arange("sample", 3),
                components=[Labels.arange("c", 2)],
                properties=block_res_2.properties,
            ),
        )

        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
        A = TensorMap(keys, [block_1, block_2])
        B = -5.1
        C = np.array([-5.1])

        tensor_sum = equistore.subtract(A, B)
        tensor_sum_array = equistore.subtract(A, C)
        tensor_result = TensorMap(keys, [block_res_1, block_res_2])

        self.assertTrue(equistore.allclose(tensor_result, tensor_sum))
        self.assertTrue(equistore.allclose(tensor_result, tensor_sum_array))

    def test_self_subtract_error(self):
        block_1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.arange("p", 2),
        )
        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0]]))
        A = TensorMap(keys, [block_1])
        B = np.ones((3, 4))

        with self.assertRaises(TypeError) as cm:
            keys = equistore.subtract(A, B)
        self.assertEqual(
            str(cm.exception),
            "B should be a TensorMap or a scalar value. ",
        )


# TODO: subtract tests with torch & torch scripting/tracing

if __name__ == "__main__":
    unittest.main()
