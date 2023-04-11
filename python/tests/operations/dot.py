import os
import unittest

import numpy as np

import equistore
from equistore import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


class TestDot(unittest.TestCase):
    def test_self_dot_no_components(self):
        tensor1 = equistore.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        tensor2 = equistore.remove_gradients(tensor1)

        dot_tensor = equistore.dot(A=tensor1, B=tensor2)
        self.assertTrue(np.all(tensor1.keys == dot_tensor.keys))

        for key, block1 in tensor1:
            dot_block = dot_tensor.block(key)

            block2 = tensor2.block(key)
            expected_values = np.dot(block1.values, block2.values.T)

            self.assertTrue(np.allclose(dot_block.values, expected_values, rtol=1e-13))
            self.assertTrue(np.all(block1.samples == dot_block.samples))
            self.assertTrue(np.all(block2.samples == dot_block.properties))

            self.assertTrue(
                len(block1.gradients_list()) == len(dot_block.gradients_list())
            )
            for parameter, gradient1 in block1.gradients():
                result_gradient = dot_block.gradient(parameter)
                self.assertTrue(np.all(gradient1.samples == result_gradient.samples))
                self.assertTrue(
                    len(gradient1.components) == len(result_gradient.components)
                )
                for c1, cres in zip(gradient1.components, result_gradient.components):
                    self.assertTrue(np.all(c1 == cres))

                self.assertTrue(len(block2.samples) == len(result_gradient.properties))
                for p1, pres in zip(block2.samples, result_gradient.properties):
                    self.assertTrue(np.all(p1 == pres))

                expected_data = gradient1.data @ block2.values.T
                self.assertTrue(
                    np.allclose(expected_data, result_gradient.data, rtol=1e-13)
                )

    def test_self_dot_components(self):
        tensor1 = equistore.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            use_numpy=True,
        )

        tensor2 = []
        n_samples = 42
        for _, block1 in tensor1:
            value2 = np.arange(n_samples * len(block1.properties)).reshape(
                n_samples, len(block1.properties)
            )

            block2 = TensorBlock(
                values=value2,
                samples=Labels(
                    ["s"],
                    np.array([[i] for i in range(n_samples)]),
                ),
                components=[],
                properties=block1.properties,
            )
            tensor2.append(block2)

        tensor2 = TensorMap(tensor1.keys, tensor2)

        dot_tensor = equistore.dot(A=tensor1, B=tensor2)
        self.assertTrue(np.all(tensor1.keys == dot_tensor.keys))

        for key, block1 in tensor1:
            block2 = tensor2.block(key)
            dot_block = dot_tensor.block(key)

            expected_values = np.dot(
                block1.values,
                block2.values.T,
            )
            self.assertTrue(np.allclose(dot_block.values, expected_values, rtol=1e-13))
            self.assertTrue(np.all(block1.samples == dot_block.samples))
            self.assertTrue(np.all(block2.samples == dot_block.properties))

            self.assertTrue(
                len(block1.gradients_list()) == len(dot_block.gradients_list())
            )
            for parameter, gradient1 in block1.gradients():
                result_gradient = dot_block.gradient(parameter)
                self.assertTrue(np.all(gradient1.samples == result_gradient.samples))
                for i in range(len(gradient1.components)):
                    self.assertTrue(
                        len(gradient1.components[i])
                        == len(result_gradient.components[i])
                    )
                for c1, cres in zip(gradient1.components, result_gradient.components):
                    self.assertTrue(np.all(c1 == cres))

                self.assertTrue(len(block2.samples) == len(result_gradient.properties))
                for p1, pres in zip(block2.samples, result_gradient.properties):
                    self.assertTrue(np.all(p1 == pres))
                expected_data = gradient1.data @ value2.T
                self.assertTrue(
                    np.allclose(expected_data, result_gradient.data, rtol=1e-13)
                )


# TODO: add tests with torch & torch scripting/tracing

if __name__ == "__main__":
    unittest.main()
