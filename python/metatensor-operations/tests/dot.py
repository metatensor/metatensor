import os
import unittest

import numpy as np

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestDot(unittest.TestCase):
    def test_self_dot_no_components(self):
        tensor_1 = metatensor.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, metatensor only supports STORED
            use_numpy=True,
        )
        tensor_2 = metatensor.remove_gradients(tensor_1)

        dot_tensor = metatensor.dot(tensor_1=tensor_1, tensor_2=tensor_2)
        self.assertTrue(np.all(tensor_1.keys == dot_tensor.keys))

        for key, block_1 in tensor_1.items():
            dot_block = dot_tensor.block(key)

            block_2 = tensor_2.block(key)
            expected_values = np.dot(block_1.values, block_2.values.T)

            self.assertTrue(np.allclose(dot_block.values, expected_values, rtol=1e-13))
            self.assertTrue(np.all(block_1.samples == dot_block.samples))
            self.assertTrue(np.all(block_2.samples == dot_block.properties))

            self.assertTrue(
                len(block_1.gradients_list()) == len(dot_block.gradients_list())
            )
            for parameter, gradient_1 in block_1.gradients():
                result_gradient = dot_block.gradient(parameter)
                self.assertTrue(np.all(gradient_1.samples == result_gradient.samples))
                self.assertTrue(
                    len(gradient_1.components) == len(result_gradient.components)
                )
                for c_1, c_res in zip(
                    gradient_1.components, result_gradient.components
                ):
                    self.assertTrue(np.all(c_1 == c_res))

                self.assertTrue(len(block_2.samples) == len(result_gradient.properties))
                for p_1, p_res in zip(block_2.samples, result_gradient.properties):
                    self.assertTrue(np.all(p_1 == p_res))

                expected_gradient_values = gradient_1.values @ block_2.values.T
                self.assertTrue(
                    np.allclose(
                        expected_gradient_values, result_gradient.values, rtol=1e-13
                    )
                )

    def test_self_dot_components(self):
        tensor_1 = metatensor.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            use_numpy=True,
        )

        tensor_2 = []
        n_samples = 42
        for block_1 in tensor_1:
            value2 = np.arange(n_samples * len(block_1.properties)).reshape(
                n_samples, len(block_1.properties)
            )

            block_2 = TensorBlock(
                values=value2,
                samples=Labels(
                    ["s"],
                    np.array([[i] for i in range(n_samples)]),
                ),
                components=[],
                properties=block_1.properties,
            )
            tensor_2.append(block_2)

        tensor_2 = TensorMap(tensor_1.keys, tensor_2)

        dot_tensor = metatensor.dot(tensor_1=tensor_1, tensor_2=tensor_2)
        self.assertTrue(np.all(tensor_1.keys == dot_tensor.keys))

        for key, block_1 in tensor_1.items():
            block_2 = tensor_2.block(key)
            dot_block = dot_tensor.block(key)

            expected_values = np.dot(
                block_1.values,
                block_2.values.T,
            )
            self.assertTrue(np.allclose(dot_block.values, expected_values, rtol=1e-13))
            self.assertTrue(np.all(block_1.samples == dot_block.samples))
            self.assertTrue(np.all(block_2.samples == dot_block.properties))

            self.assertTrue(
                len(block_1.gradients_list()) == len(dot_block.gradients_list())
            )
            for parameter, gradient_1 in block_1.gradients():
                result_gradient = dot_block.gradient(parameter)
                self.assertTrue(np.all(gradient_1.samples == result_gradient.samples))
                for i in range(len(gradient_1.components)):
                    self.assertTrue(
                        len(gradient_1.components[i])
                        == len(result_gradient.components[i])
                    )
                for c1, cres in zip(gradient_1.components, result_gradient.components):
                    self.assertTrue(np.all(c1 == cres))

                self.assertTrue(len(block_2.samples) == len(result_gradient.properties))
                for p1, pres in zip(block_2.samples, result_gradient.properties):
                    self.assertTrue(np.all(p1 == pres))
                expected_gradient_values = gradient_1.values @ value2.T
                self.assertTrue(
                    np.allclose(
                        expected_gradient_values, result_gradient.values, rtol=1e-13
                    )
                )


# TODO: add tests with torch & torch scripting/tracing

if __name__ == "__main__":
    unittest.main()
