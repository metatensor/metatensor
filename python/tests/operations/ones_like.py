import os
import unittest

import numpy as np

import equistore


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


class TestOnes_like(unittest.TestCase):
    def test_ones_like_nocomponent(self):
        tensor = equistore.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        ones_tensor = equistore.ones_like(tensor)

        self.assertTrue(np.all(tensor.keys == ones_tensor.keys))
        for key, ones_block in ones_tensor:
            block = tensor.block(key)
            self.assertTrue(np.all(ones_block.samples == block.samples))
            self.assertTrue(np.all(ones_block.properties == block.properties))
            self.assertEqual(len(ones_block.components), len(block.components))
            self.assertTrue(
                np.all(
                    [
                        np.all(ones_block.components[i] == block.components[i])
                        for i in range(len(block.components))
                    ]
                )
            )
            self.assertTrue(np.allclose(ones_block.values, np.ones_like(block.values)))
            for ones_parameter, ones_gradient in ones_block.gradients():
                gradient = block.gradient(ones_parameter)
                self.assertTrue(np.all(ones_gradient.samples == gradient.samples))
                self.assertEqual(
                    len(ones_gradient.components), len(gradient.components)
                )
                self.assertTrue(
                    np.all(
                        [
                            np.all(
                                ones_gradient.components[i] == gradient.components[i]
                            )
                            for i in range(len(gradient.components))
                        ]
                    )
                )
                self.assertTrue(
                    np.allclose(ones_gradient.values, np.ones_like(gradient.values))
                )

    def test_ones_component(self):
        tensor = equistore.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        ones_tensor = equistore.ones_like(tensor)
        ones_tensor_positions = equistore.ones_like(tensor, parameters="positions")

        self.assertTrue(np.all(tensor.keys == ones_tensor.keys))
        self.assertTrue(np.all(tensor.keys == ones_tensor_positions.keys))
        for key, ones_block in ones_tensor:
            block = tensor.block(key)
            ones_block_pos = ones_tensor_positions.block(key)
            self.assertTrue(np.all(ones_block.samples == block.samples))
            self.assertTrue(np.all(ones_block.properties == block.properties))
            self.assertTrue(np.allclose(ones_block.values, np.ones_like(block.values)))

            self.assertTrue(np.all(ones_block_pos.samples == block.samples))
            self.assertTrue(np.all(ones_block_pos.properties == block.properties))
            self.assertTrue(
                np.allclose(ones_block_pos.values, np.ones_like(block.values))
            )

            self.assertTrue(ones_block.gradients_list() == block.gradients_list())
            for ones_parameter, ones_gradient in ones_block.gradients():
                gradient = block.gradient(ones_parameter)
                self.assertTrue(np.all(ones_gradient.samples == gradient.samples))
                self.assertEqual(
                    len(ones_gradient.components), len(gradient.components)
                )
                self.assertTrue(
                    np.all(
                        [
                            np.all(
                                ones_gradient.components[i] == gradient.components[i]
                            )
                            for i in range(len(gradient.components))
                        ]
                    )
                )
                self.assertTrue(
                    np.allclose(ones_gradient.values, np.ones_like(gradient.values))
                )
            self.assertTrue(ones_block_pos.gradients_list() == ["positions"])
            for ones_parameter_pos, ones_gradient_pos in ones_block_pos.gradients():
                gradient = block.gradient(ones_parameter_pos)
                self.assertTrue(np.all(ones_gradient_pos.samples == gradient.samples))
                self.assertEqual(
                    len(ones_gradient_pos.components), len(gradient.components)
                )
                self.assertTrue(
                    np.all(
                        [
                            np.all(
                                ones_gradient_pos.components[i]
                                == gradient.components[i]
                            )
                            for i in range(len(gradient.components))
                        ]
                    )
                )
                self.assertTrue(
                    np.allclose(ones_gradient_pos.values, np.ones_like(gradient.values))
                )

    def test_ones_error(self):
        tensor = equistore.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )

        with self.assertRaises(ValueError) as cm:
            tensor = equistore.ones_like(tensor, parameters=["positions", "err"])
        self.assertEqual(
            str(cm.exception),
            "The requested parameter 'err' in ones_like_block "
            "is not a valid parameterfor the TensorBlock",
        )


# TODO: add tests with torch & torch scripting/tracing

if __name__ == "__main__":
    unittest.main()
