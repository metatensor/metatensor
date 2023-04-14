import os
import unittest

import equistore


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestRemoveGradients(unittest.TestCase):
    def test_remove_everything(self):
        tensor = equistore.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )

        self.assertEqual(
            set(tensor.block(0).gradients_list()), set(["cell", "positions"])
        )

        tensor = equistore.remove_gradients(tensor)
        self.assertEqual(tensor.block(0).gradients_list(), [])

    def test_remove_subset(self):
        tensor = equistore.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )

        self.assertEqual(
            set(tensor.block(0).gradients_list()), set(["cell", "positions"])
        )

        tensor = equistore.remove_gradients(tensor, ["positions"])
        self.assertEqual(tensor.block(0).gradients_list(), ["cell"])


if __name__ == "__main__":
    unittest.main()
