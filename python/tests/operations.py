import os
import unittest

import numpy as np
from equistore import TensorMap
import equistore.operations as fn
import equistore.io
from utils import compare_blocks

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


class TestDot(unittest.TestCase):
    def test_self_dot_no_components(self):
        tensor1 = equistore.io.load(
            os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
            use_numpy=True,
        )
        tensor2 = fn.remove_gradients(tensor1)
        dot_blocks = []
        for key, block1 in tensor1:
            block2 = tensor2.block(key)
            result_block = fn._dot_block(block1, block2)
            dot_blocks.append(result_block)
            expected_values = np.dot(block1.values, block2.values.T)
            self.assertTrue(
                np.allclose(result_block.values, expected_values, rtol=1e-13)
            )
            self.assertTrue(np.all(block1.samples == result_block.samples))
            self.assertTrue(np.all(block2.samples == result_block.properties))

            self.assertTrue(
                len(block1.gradients_list()) == len(result_block.gradients_list())
            )
            for parameter, gradient1 in block1.gradients():
                result_gradient = result_block.gradient(parameter)
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
        expected_tensor = TensorMap(tensor1.keys, dot_blocks)

        dot_tensor = fn.dot(tensor1=tensor1, tensor2=tensor2)
        self.assertTrue(np.all(expected_tensor.keys == dot_tensor.keys))
        for key, expected_block in expected_tensor:

            comparing_dict = compare_blocks(expected_block, dot_tensor.block(key))
            self.assertTrue(comparing_dict["general"])


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
