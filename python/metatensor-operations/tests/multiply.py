import os
import unittest

import numpy as np

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestMultiply(unittest.TestCase):
    def test_self_multiply_tensors_nogradient(self):
        block_1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        block_2 = TensorBlock(
            values=np.array([[1, 2], [3, 4], [5, 6]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        block_3 = TensorBlock(
            values=np.array([[1.5, 2.1], [6.7, 10.2]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        block_4 = TensorBlock(
            values=np.array([[10, 200.8], [3.76, 4.432], [545, 26]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.range("p", 2),
        )

        block_res_1 = TensorBlock(
            values=np.array([[1.5, 4.2], [20.1, 51.0]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        block_res_2 = TensorBlock(
            values=np.array([[10.0, 401.6], [11.28, 17.728], [2725.0, 156.0]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
        A = TensorMap(keys, [block_1, block_2])
        B = TensorMap(keys, [block_3, block_4])
        A_copy = A.copy()
        B_copy = B.copy()
        tensor_sum = metatensor.multiply(A, B)
        tensor_result = TensorMap(keys, [block_res_1, block_res_2])

        self.assertTrue(metatensor.allclose(tensor_result, tensor_sum))
        # Check not modified in place
        self.assertTrue(metatensor.allclose(A, A_copy))
        self.assertTrue(metatensor.allclose(B, B_copy))

    def test_self_multiply_tensors_gradient(self):
        block_1 = TensorBlock(
            values=np.array([[14, 24], [43, 45]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        block_1.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array([[[6, 1], [7, 2]], [[8, 3], [9, 4]]]),
                samples=Labels.range("sample", 2),
                components=[Labels.range("c", 2)],
                properties=block_1.properties,
            ),
        )

        block_2 = TensorBlock(
            values=np.array([[15, 25], [53, 54], [55, 65]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        block_2.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array(
                    [[[10, 11], [12, 13]], [[14, 15], [10, 11]], [[12, 13], [14, 15]]]
                ),
                samples=Labels.range("sample", 3),
                components=[Labels.range("c", 2)],
                properties=block_2.properties,
            ),
        )

        block_3 = TensorBlock(
            values=np.array([[1.45, 2.41], [6.47, 10.42]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        block_3.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array([[[1, 0.1], [2, 0.2]], [[3, 0.3], [4.5, 0.4]]]),
                samples=Labels.range("sample", 2),
                components=[Labels.range("c", 2)],
                properties=block_3.properties,
            ),
        )

        block_4 = TensorBlock(
            values=np.array([[105, 200.58], [3.756, 4.4325], [545.5, 26.05]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        block_4.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array(
                    [
                        [[1.0, 1.1], [1.2, 1.3]],
                        [[1.4, 1.5], [1.0, 1.1]],
                        [[1.2, 1.3], [1.4, 1.5]],
                    ]
                ),
                samples=Labels.range("sample", 3),
                components=[Labels.range("c", 2)],
                properties=block_4.properties,
            ),
        )

        block_res_1 = TensorBlock(
            values=np.array([[20.3, 57.84], [278.21, 468.9]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        block_res_1.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array(
                    [[[22.7, 4.81], [38.15, 9.62]], [[180.76, 44.76], [251.73, 59.68]]]
                ),
                samples=Labels.range("sample", 2),
                components=[Labels.range("c", 2)],
                properties=block_res_1.properties,
            ),
        )

        block_res_2 = TensorBlock(
            values=np.array([[1575.0, 5014.5], [199.068, 239.355], [30002.5, 1693.25]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        block_res_2.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array(
                    [
                        [[1065.0, 2233.88], [1278.0, 2640.04]],
                        [[126.784, 147.4875], [90.56, 108.1575]],
                        [[6612.0, 423.15], [7714.0, 488.25]],
                    ]
                ),
                samples=Labels.range("sample", 3),
                components=[Labels.range("c", 2)],
                properties=block_res_2.properties,
            ),
        )

        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
        A = TensorMap(keys, [block_1, block_2])
        B = TensorMap(keys, [block_3, block_4])
        A_copy = A.copy()
        B_copy = B.copy()
        tensor_sum = metatensor.multiply(A, B)
        tensor_result = TensorMap(keys, [block_res_1, block_res_2])

        self.assertTrue(metatensor.allclose(tensor_result, tensor_sum))
        # Check not modified in place
        self.assertTrue(metatensor.equal(A, A_copy))
        self.assertTrue(metatensor.equal(B, B_copy))

    def test_self_multiply_scalar_gradient(self):
        block_1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        block_1.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array([[[6, 1], [7, 2]], [[8, 3], [9, 4]]]),
                samples=Labels.range("sample", 2),
                components=[Labels.range("c", 2)],
                properties=block_1.properties,
            ),
        )

        block_2 = TensorBlock(
            values=np.array([[11, 12], [13, 14], [15, 16]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        block_2.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array(
                    [[[10, 11], [12, 13]], [[14, 15], [10, 11]], [[12, 13], [14, 15]]]
                ),
                samples=Labels.range("sample", 3),
                components=[Labels.range("c", 2)],
                properties=block_2.properties,
            ),
        )

        block_res_1 = TensorBlock(
            values=np.array([[5.1, 10.2], [15.3, 25.5]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        block_res_1.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array(
                    [[[30.6, 5.1], [35.7, 10.2]], [[40.8, 15.3], [45.9, 20.4]]]
                ),
                samples=Labels.range("sample", 2),
                components=[Labels.range("c", 2)],
                properties=block_res_1.properties,
            ),
        )

        block_res_2 = TensorBlock(
            values=np.array([[56.1, 61.2], [66.3, 71.4], [76.5, 81.6]]),
            samples=Labels(["s"], np.array([[0], [2], [7]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        block_res_2.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=np.array(
                    [
                        [[51.0, 56.1], [61.2, 66.3]],
                        [[71.4, 76.5], [51.0, 56.1]],
                        [[61.2, 66.3], [71.4, 76.5]],
                    ]
                ),
                samples=Labels.range("sample", 3),
                components=[Labels.range("c", 2)],
                properties=block_res_2.properties,
            ),
        )

        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
        A = TensorMap(keys, [block_1, block_2])
        B = 5.1

        tensor_sum = metatensor.multiply(A, B)
        tensor_result = TensorMap(keys, [block_res_1, block_res_2])

        self.assertTrue(metatensor.allclose(tensor_result, tensor_sum))

    def test_self_multiply_error(self):
        block_1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["s"], np.array([[0], [2]])),
            components=[],
            properties=Labels.range("p", 2),
        )
        keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0]]))
        A = TensorMap(keys, [block_1])
        B = np.ones((3, 4))

        with self.assertRaises(TypeError) as cm:
            keys = metatensor.multiply(A, B)
        self.assertEqual(
            str(cm.exception),
            "B should be a TensorMap or a scalar value",
        )


# TODO: multiply tests with torch & torch scripting/tracing

if __name__ == "__main__":
    unittest.main()
