import os
import unittest

import equistore.io
import equistore.operations as fn


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


class TestRemoveGradients(unittest.TestCase):
    def test_remove_everything(self):
        tensor = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )

        self.assertEqual(tensor.block(0).gradients_list(), ["cell", "positions"])

        tensor = fn.remove_gradients(tensor)
        self.assertEqual(tensor.block(0).gradients_list(), [])

    def test_remove_subset(self):
        tensor = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            # the npz is using DEFLATE compression, equistore only supports STORED
            use_numpy=True,
        )

        self.assertEqual(tensor.block(0).gradients_list(), ["cell", "positions"])

        tensor = fn.remove_gradients(tensor, ["positions"])
        self.assertEqual(tensor.block(0).gradients_list(), ["cell"])


if __name__ == "__main__":
    unittest.main()
