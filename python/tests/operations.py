import os
import unittest

import numpy as np

import equistore.functions as fn
import equistore.io

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestNormalize(unittest.TestCase):
    def test_normalize_no_components(self):
        data = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            use_numpy=True,
        )

        with self.assertRaises(ValueError) as cm:
            _ = fn.normalize_by_sample(data)

        self.assertEqual(
            str(cm.exception),
            "normalization of gradients w.r.t. 'cell' is not yet implemented",
        )

        data = fn.remove_gradients(data, remove=["cell"])
        normalized = fn.normalize_by_sample(data)

        for _, block in normalized:
            norm = np.linalg.norm(block.values, axis=-1)
            self.assertTrue(np.allclose(norm, np.ones_like(norm), rtol=1e-16))

        # TODO: add tests for gradients with finite differences

    def test_normalize_components(self):
        data = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
            use_numpy=True,
        )

        with self.assertRaises(ValueError) as cm:
            _ = fn.normalize_by_sample(data)

        self.assertEqual(
            str(cm.exception),
            "normalization of equivariant tensors is not yet implemented",
        )


# TODO: add tests with torch & torch scripting/tracing

if __name__ == "__main__":
    unittest.main()
