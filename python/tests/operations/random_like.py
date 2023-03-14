import os
import unittest

import numpy as np

import equistore


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


class TestRandomLike(unittest.TestCase):
    def test_random_uniform_like_nocomponent(self):
        tensor = equistore.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        random_uniform_tensor = equistore.random_uniform_like(tensor)

        self.assertTrue(np.all(tensor.keys == random_uniform_tensor.keys))
        for key, random_uniform_block in random_uniform_tensor:
            block = tensor.block(key)
            self.assertTrue(np.all(random_uniform_block.samples == block.samples))
            self.assertTrue(np.all(random_uniform_block.properties == block.properties))
            self.assertEqual(
                len(random_uniform_block.components), len(block.components)
            )
            self.assertTrue(
                np.all(
                    [
                        np.all(
                            random_uniform_block.components[i] == block.components[i]
                        )
                        for i in range(len(block.components))
                    ]
                )
            )
            self.assertTrue(np.all(random_uniform_block.values >= 0))
            self.assertTrue(np.all(random_uniform_block.values <= 1))
            self.assertTrue(
                np.isclose(random_uniform_block.values.mean(), 0.5, atol=0.1)
            )
            for ones_parameter, ones_gradient in random_uniform_block.gradients():
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
                self.assertTrue(np.all(ones_gradient.data >= 0))
                self.assertTrue(np.all(ones_gradient.data <= 1))
                self.assertTrue(np.isclose(ones_gradient.data.mean(), 0.5, atol=0.1))

    def test_random_uniform_like_component(self):
        tensor = equistore.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )
        rand_tensor = equistore.random_uniform_like(tensor)
        rand_tensor_positions = equistore.random_uniform_like(
            tensor, parameters="positions"
        )

        self.assertTrue(np.all(tensor.keys == rand_tensor.keys))
        self.assertTrue(np.all(tensor.keys == rand_tensor_positions.keys))
        for key, rand_block in rand_tensor:
            block = tensor.block(key)
            rand_block_pos = rand_tensor_positions.block(key)
            self.assertTrue(np.all(rand_block.samples == block.samples))
            self.assertTrue(np.all(rand_block.properties == block.properties))
            self.assertTrue(np.all(rand_block.values >= 0))
            self.assertTrue(np.all(rand_block.values <= 1))
            self.assertTrue(np.isclose(rand_block.values.mean(), 0.5, atol=0.1))

            self.assertTrue(np.all(rand_block_pos.samples == block.samples))
            self.assertTrue(np.all(rand_block_pos.properties == block.properties))
            self.assertTrue(np.all(rand_block_pos.values >= 0))
            self.assertTrue(np.all(rand_block_pos.values <= 1))
            self.assertTrue(np.isclose(rand_block_pos.values.mean(), 0.5, atol=0.1))

            self.assertTrue(rand_block.gradients_list() == block.gradients_list())
            for rand_parameter, rand_gradient in rand_block.gradients():
                gradient = block.gradient(rand_parameter)
                self.assertTrue(np.all(rand_gradient.samples == gradient.samples))
                self.assertEqual(
                    len(rand_gradient.components), len(gradient.components)
                )
                self.assertTrue(
                    np.all(
                        [
                            np.all(
                                rand_gradient.components[i] == gradient.components[i]
                            )
                            for i in range(len(gradient.components))
                        ]
                    )
                )

                self.assertTrue(np.all(rand_gradient.data >= 0))
                self.assertTrue(np.all(rand_gradient.data <= 1))
                self.assertTrue(np.isclose(rand_gradient.data.mean(), 0.5, atol=0.1))

            self.assertTrue(rand_block_pos.gradients_list() == ["positions"])
            for rand_parameter_pos, rand_gradient_pos in rand_block_pos.gradients():
                gradient = block.gradient(rand_parameter_pos)
                self.assertTrue(np.all(rand_gradient_pos.samples == gradient.samples))
                self.assertEqual(
                    len(rand_gradient_pos.components), len(gradient.components)
                )
                self.assertTrue(
                    np.all(
                        [
                            np.all(
                                rand_gradient_pos.components[i]
                                == gradient.components[i]
                            )
                            for i in range(len(gradient.components))
                        ]
                    )
                )
                self.assertTrue(np.all(rand_gradient_pos.data >= 0))
                self.assertTrue(np.all(rand_gradient_pos.data <= 1))
                self.assertTrue(
                    np.isclose(rand_gradient_pos.data.mean(), 0.5, atol=0.1)
                )

    def test_random_uniform_like_error(self):
        tensor = equistore.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )

        with self.assertRaises(ValueError) as cm:
            tensor = equistore.random_uniform_like(
                tensor, parameters=["positions", "err"]
            )
        self.assertEqual(
            str(cm.exception),
            "The requested parameter 'err' in ones_like_block "
            "is not a valid parameterfor the TensorBlock",
        )


# TODO: add tests with torch & torch scripting/tracing

if __name__ == "__main__":
    unittest.main()
