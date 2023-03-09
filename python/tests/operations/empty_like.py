import os
import unittest

import numpy as np

import equistore


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


class TestEmpty_like(unittest.TestCase):
    def test_empty_like_nocomponent(self):
        tensor = equistore.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        empty_tensor = equistore.empty_like(tensor)

        self.assertTrue(np.all(tensor.keys == empty_tensor.keys))
        for key, empty_block in empty_tensor:
            block = tensor.block(key)
            self.assertTrue(np.all(empty_block.samples == block.samples))
            self.assertTrue(np.all(empty_block.properties == block.properties))
            self.assertEqual(len(empty_block.components), len(block.components))
            self.assertTrue(
                np.all(
                    [
                        np.all(empty_block.components[i] == block.components[i])
                        for i in range(len(block.components))
                    ]
                )
            )
            self.assertEqual(
                empty_block.values.shape, np.empty_like(block.values).shape
            )

            for empty_parameter, empty_gradient in empty_block.gradients():
                gradient = block.gradient(empty_parameter)
                self.assertTrue(np.all(empty_gradient.samples == gradient.samples))
                self.assertEqual(
                    len(empty_gradient.components), len(gradient.components)
                )
                self.assertTrue(
                    np.all(
                        [
                            np.all(
                                empty_gradient.components[i] == gradient.components[i]
                            )
                            for i in range(len(gradient.components))
                        ]
                    )
                )
                self.assertEqual(
                    empty_gradient.data.shape, np.empty_like(gradient.data).shape
                )

    def test_empty_component(self):
        tensor = equistore.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        empty_tensor = equistore.empty_like(tensor)
        empty_tensor_positions = equistore.empty_like(tensor, parameters="positions")

        self.assertTrue(np.all(tensor.keys == empty_tensor.keys))
        self.assertTrue(np.all(tensor.keys == empty_tensor_positions.keys))
        for key, empty_block in empty_tensor:
            block = tensor.block(key)
            empty_block_pos = empty_tensor_positions.block(key)
            self.assertTrue(np.all(empty_block.samples == block.samples))
            self.assertTrue(np.all(empty_block.properties == block.properties))

            self.assertTrue(np.all(empty_block_pos.samples == block.samples))
            self.assertTrue(np.all(empty_block_pos.properties == block.properties))
            self.assertEqual(
                empty_block_pos.values.shape, np.empty_like(block.values).shape
            )

            self.assertTrue(empty_block.gradients_list() == block.gradients_list())
            for empty_parameter, empty_gradient in empty_block.gradients():
                gradient = block.gradient(empty_parameter)
                self.assertTrue(np.all(empty_gradient.samples == gradient.samples))
                self.assertEqual(
                    len(empty_gradient.components), len(gradient.components)
                )
                self.assertTrue(
                    np.all(
                        [
                            np.all(
                                empty_gradient.components[i] == gradient.components[i]
                            )
                            for i in range(len(gradient.components))
                        ]
                    )
                )
                self.assertEqual(
                    empty_gradient.data.shape, np.empty_like(gradient.data).shape
                )

            self.assertTrue(empty_block_pos.gradients_list() == ["positions"])
            for empty_parameter_pos, empty_gradient_pos in empty_block_pos.gradients():
                gradient = block.gradient(empty_parameter_pos)
                self.assertTrue(np.all(empty_gradient_pos.samples == gradient.samples))
                self.assertEqual(
                    len(empty_gradient_pos.components), len(gradient.components)
                )
                self.assertTrue(
                    np.all(
                        [
                            np.all(
                                empty_gradient_pos.components[i]
                                == gradient.components[i]
                            )
                            for i in range(len(gradient.components))
                        ]
                    )
                )
                self.assertEqual(
                    empty_gradient_pos.data.shape, np.empty_like(gradient.data).shape
                )

    def test_empty_error(self):
        tensor = equistore.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )

        with self.assertRaises(ValueError) as cm:
            tensor = equistore.empty_like(tensor, parameters=["positions", "err"])
        self.assertEqual(
            str(cm.exception),
            "The requested parameter 'err' in empty_like_block "
            "is not a valid parameterfor the TensorBlock",
        )


# TODO: add tests with torch & torch scripting/tracing

if __name__ == "__main__":
    unittest.main()
