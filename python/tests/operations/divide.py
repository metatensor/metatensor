import os
import unittest

import numpy as np

import equistore.operations as fn
from equistore import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


class TestDivide(unittest.TestCase):
    def test_self_divide_tensors_nogradient(self):
        block_1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_2 = TensorBlock(
            values=np.array([[1, 2], [3, 4], [5, 6]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_3 = TensorBlock(
            values=np.array([[1.5, 2.1], [6.7, 10.2]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_4 = TensorBlock(
            values=np.array([[10, 200.8], [3.76, 4.432], [545, 26]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )

        block_res1 = TensorBlock(
            values=block_1.values[:] / block_3.values[:],
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_res2 = TensorBlock(
            values=block_2.values[:] / block_4.values[:],
            samples=Labels(["samples"], np.array([[0], [2], [7]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        keys = Labels(
            names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]], dtype=np.int32)
        )
        A = TensorMap(keys, [block_1, block_2])
        B = TensorMap(keys, [block_3, block_4])
        tensor_sum = fn.divide(A, B)
        tensor_result = TensorMap(keys, [block_res1, block_res2])

        self.assertTrue(fn.allclose(tensor_result, tensor_sum))

    def test_self_divide_tensors_gradient(self):
        block_1 = TensorBlock(
            values=np.array([[14, 24], [43, 45]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_1.add_gradient(
            "parameter",
            data=np.array([[[6, 1], [7, 2]], [[8, 3], [9, 4]]]),
            samples=Labels(
                ["sample", "positions"], np.array([[0, 1], [1, 1]], dtype=np.int32)
            ),
            components=[
                Labels(["components"], np.array([[0], [1]], dtype=np.int32)),
            ],
        )

        block_2 = TensorBlock(
            values=np.array([[15, 25], [53, 54], [55, 65]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_2.add_gradient(
            "parameter",
            data=np.array(
                [[[10, 11], [12, 13]], [[14, 15], [10, 11]], [[12, 13], [14, 15]]]
            ),
            samples=Labels(
                ["sample", "positions"],
                np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int32),
            ),
            components=[
                Labels(["components"], np.array([[0], [1]], dtype=np.int32)),
            ],
        )

        block_3 = TensorBlock(
            values=np.array([[1.45, 2.41], [6.47, 10.42]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_3.add_gradient(
            "parameter",
            data=np.array([[[1, 0.1], [2, 0.2]], [[3, 0.3], [4.5, 0.4]]]),
            samples=Labels(
                ["sample", "positions"], np.array([[0, 1], [1, 1]], dtype=np.int32)
            ),
            components=[
                Labels(["components"], np.array([[0], [1]], dtype=np.int32)),
            ],
        )
        block_4 = TensorBlock(
            values=np.array([[105, 200.58], [3.756, 4.4325], [545.5, 26.05]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_4.add_gradient(
            "parameter",
            data=np.array(
                [
                    [[1.0, 1.1], [1.2, 1.3]],
                    [[1.4, 1.5], [1.0, 1.1]],
                    [[1.2, 1.3], [1.4, 1.5]],
                ]
            ),
            samples=Labels(
                ["sample", "positions"],
                np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int32),
            ),
            components=[
                Labels(["components"], np.array([[0], [1]], dtype=np.int32)),
            ],
        )

        block_res1 = TensorBlock(
            values=np.array([[9.65517241, 9.95850622], [6.64605873, 4.31861804]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_res1.add_gradient(
            "parameter",
            data=np.array(
                [
                    [
                        [-2.52080856e00, 1.72173344e-03],
                        [-8.48989298e00, 3.44346688e-03],
                    ],
                    [[-1.84515861, 0.16357146], [-3.23141643, 0.21809528]],
                ]
            ),
            samples=Labels(
                ["sample", "positions"], np.array([[0, 1], [1, 1]], dtype=np.int32)
            ),
            components=[
                Labels(["components"], np.array([[0], [1]], dtype=np.int32)),
            ],
        )
        block_res2 = TensorBlock(
            values=np.array(
                [
                    [0.14285714, 0.12463855],
                    [14.11075612, 12.18274112],
                    [0.10082493, 2.49520154],
                ]
            ),
            samples=Labels(["samples"], np.array([[0], [2], [7]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_res2.add_gradient(
            "parameter",
            data=np.array(
                [
                    [[0.09387755, 0.05415743], [0.11265306, 0.06400424]],
                    [[-1.53223072, -0.73866028], [-1.09445051, -0.5416842]],
                    [[0.02177637, 0.37451969], [0.02540577, 0.43213811]],
                ]
            ),
            samples=Labels(
                ["sample", "positions"],
                np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int32),
            ),
            components=[
                Labels(["components"], np.array([[0], [1]], dtype=np.int32)),
            ],
        )
        keys = Labels(
            names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]], dtype=np.int32)
        )
        A = TensorMap(keys, [block_1, block_2])
        B = TensorMap(keys, [block_3, block_4])
        tensor_sum = fn.divide(A, B)
        tensor_result = TensorMap(keys, [block_res1, block_res2])
        self.assertTrue(
            fn.allclose(
                tensor_result,
                tensor_sum,
                atol=1e-8,
            )
        )

    def test_self_divide_scalar_gradient(self):
        block_1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_1.add_gradient(
            "parameter",
            data=np.array([[[6, 1], [7, 2]], [[8, 3], [9, 4]]]),
            samples=Labels(
                ["sample", "positions"], np.array([[0, 1], [1, 1]], dtype=np.int32)
            ),
            components=[
                Labels(["components"], np.array([[0], [1]], dtype=np.int32)),
            ],
        )
        block_2 = TensorBlock(
            values=np.array([[11, 12], [13, 14], [15, 16]]),
            samples=Labels(["samples"], np.array([[0], [2], [7]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_2.add_gradient(
            "parameter",
            data=np.array(
                [[[10, 11], [12, 13]], [[14, 15], [10, 11]], [[12, 13], [14, 15]]]
            ),
            samples=Labels(
                ["sample", "positions"],
                np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int32),
            ),
            components=[
                Labels(["components"], np.array([[0], [1]], dtype=np.int32)),
            ],
        )

        block_res1 = TensorBlock(
            values=np.array([[0.19607843, 0.39215686], [0.58823529, 0.98039216]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_res1.add_gradient(
            "parameter",
            data=np.array(
                [
                    [[1.17647059, 0.19607843], [1.37254902, 0.39215686]],
                    [[1.56862745, 0.58823529], [1.76470588, 0.78431373]],
                ]
            ),
            samples=Labels(
                ["sample", "positions"], np.array([[0, 1], [1, 1]], dtype=np.int32)
            ),
            components=[
                Labels(["components"], np.array([[0], [1]], dtype=np.int32)),
            ],
        )
        block_res2 = TensorBlock(
            values=np.array(
                [
                    [2.15686275, 2.35294118],
                    [2.54901961, 2.74509804],
                    [2.94117647, 3.1372549],
                ]
            ),
            samples=Labels(["samples"], np.array([[0], [2], [7]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_res2.add_gradient(
            "parameter",
            data=np.array(
                [
                    [[1.96078431, 2.15686275], [2.35294118, 2.54901961]],
                    [[2.74509804, 2.94117647], [1.96078431, 2.15686275]],
                    [[2.35294118, 2.54901961], [2.74509804, 2.94117647]],
                ]
            ),
            samples=Labels(
                ["sample", "positions"],
                np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int32),
            ),
            components=[
                Labels(["components"], np.array([[0], [1]], dtype=np.int32)),
            ],
        )

        keys = Labels(
            names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]], dtype=np.int32)
        )
        A = TensorMap(keys, [block_1, block_2])
        B = 5.1
        C = np.array([5.1])

        tensor_sum = fn.divide(A, B)
        tensor_sum_array = fn.divide(A, C)
        tensor_result = TensorMap(keys, [block_res1, block_res2])

        self.assertTrue(fn.allclose(tensor_result, tensor_sum, rtol=1e-8))
        self.assertTrue(fn.allclose(tensor_result, tensor_sum_array, rtol=1e-8))

    def test_self_divide_error(self):
        block_1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        keys = Labels(
            names=["key_1", "key_2"], values=np.array([[0, 0]], dtype=np.int32)
        )
        A = TensorMap(keys, [block_1])
        B = np.ones((3, 4))

        with self.assertRaises(TypeError) as cm:
            keys = fn.divide(A, B)
        self.assertEqual(
            str(cm.exception),
            "B should be a TensorMap or a scalar value. ",
        )


# TODO: divide tests with torch & torch scripting/tracing

if __name__ == "__main__":
    unittest.main()
