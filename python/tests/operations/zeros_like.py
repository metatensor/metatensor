import os
import unittest

import numpy as np

import equistore


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


class TestZeros_like(unittest.TestCase):
    def test_zeros_like_nocomponent(self):
        tensor = equistore.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        zeros_tensor = equistore.zeros_like(tensor)

        self.assertTrue(np.all(tensor.keys == zeros_tensor.keys))
        for key, zeros_block in zeros_tensor:
            block = tensor.block(key)
            self.assertTrue(np.all(zeros_block.samples == block.samples))
            self.assertTrue(np.all(zeros_block.properties == block.properties))
            self.assertEqual(len(zeros_block.components), len(block.components))
            self.assertTrue(
                np.all(
                    [
                        np.all(zeros_block.components[i] == block.components[i])
                        for i in range(len(block.components))
                    ]
                )
            )
            self.assertTrue(
                np.allclose(zeros_block.values, np.zeros_like(block.values))
            )
            for zeros_parameter, zeros_gradient in zeros_block.gradients():
                gradient = block.gradient(zeros_parameter)
                self.assertTrue(np.all(zeros_gradient.samples == gradient.samples))
                self.assertEqual(
                    len(zeros_gradient.components), len(gradient.components)
                )
                self.assertTrue(
                    np.all(
                        [
                            np.all(
                                zeros_gradient.components[i] == gradient.components[i]
                            )
                            for i in range(len(gradient.components))
                        ]
                    )
                )
                self.assertTrue(
                    np.allclose(zeros_gradient.data, np.zeros_like(gradient.data))
                )

    def test_zeros_component(self):
        tensor = equistore.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        zeros_tensor = equistore.zeros_like(tensor)
        zeros_tensor_positions = equistore.zeros_like(tensor, parameters="positions")

        self.assertTrue(np.all(tensor.keys == zeros_tensor.keys))
        self.assertTrue(np.all(tensor.keys == zeros_tensor_positions.keys))
        for key, zeros_block in zeros_tensor:
            block = tensor.block(key)
            zeros_block_pos = zeros_tensor_positions.block(key)
            self.assertTrue(np.all(zeros_block.samples == block.samples))
            self.assertTrue(np.all(zeros_block.properties == block.properties))
            self.assertTrue(
                np.allclose(zeros_block.values, np.zeros_like(block.values))
            )

            self.assertTrue(np.all(zeros_block_pos.samples == block.samples))
            self.assertTrue(np.all(zeros_block_pos.properties == block.properties))
            self.assertTrue(
                np.allclose(zeros_block_pos.values, np.zeros_like(block.values))
            )

            self.assertTrue(zeros_block.gradients_list() == block.gradients_list())
            for zeros_parameter, zeros_gradient in zeros_block.gradients():
                gradient = block.gradient(zeros_parameter)
                self.assertTrue(np.all(zeros_gradient.samples == gradient.samples))
                self.assertEqual(
                    len(zeros_gradient.components), len(gradient.components)
                )
                self.assertTrue(
                    np.all(
                        [
                            np.all(
                                zeros_gradient.components[i] == gradient.components[i]
                            )
                            for i in range(len(gradient.components))
                        ]
                    )
                )
                self.assertTrue(
                    np.allclose(zeros_gradient.data, np.zeros_like(gradient.data))
                )
            self.assertTrue(zeros_block_pos.gradients_list() == ["positions"])
            for zeros_parameter_pos, zeros_gradient_pos in zeros_block_pos.gradients():
                gradient = block.gradient(zeros_parameter_pos)
                self.assertTrue(np.all(zeros_gradient_pos.samples == gradient.samples))
                self.assertEqual(
                    len(zeros_gradient_pos.components), len(gradient.components)
                )
                self.assertTrue(
                    np.all(
                        [
                            np.all(
                                zeros_gradient_pos.components[i]
                                == gradient.components[i]
                            )
                            for i in range(len(gradient.components))
                        ]
                    )
                )
                self.assertTrue(
                    np.allclose(zeros_gradient_pos.data, np.zeros_like(gradient.data))
                )

    def test_zeros_error(self):
        tensor = equistore.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )

        with self.assertRaises(ValueError) as cm:
            tensor = equistore.zeros_like(tensor, parameters=["positions", "err"])
        self.assertEqual(
            str(cm.exception),
            "requested gradient 'err' in zeros_like is not defined in this tensor",
        )


# TODO: add tests with torch & torch scripting/tracing

if __name__ == "__main__":
    unittest.main()
