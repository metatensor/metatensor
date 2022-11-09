import os
import unittest

import numpy as np
from utils import compare_blocks

import equistore.operations as fn
from equistore import Labels, TensorBlock, TensorMap

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


class TestSolve(unittest.TestCase):
    def test_self_solve_nograd(self):
        block_1 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_2 = TensorBlock(
            values=np.array([[1, 2], [3, 5]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
        )
        block_3 = TensorBlock(
            values=np.array([[1], [2]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )
        block_4 = TensorBlock(
            values=np.array([[2], [4]]),
            samples=Labels(["samples"], np.array([[0], [2]], dtype=np.int32)),
            components=[],
            properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
        )
        keys = Labels(
            names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]], dtype=np.int32)
        )
        X = TensorMap(keys, [block_1, block_2])
        Y = TensorMap(keys, [block_3, block_4])
        w = fn.solve(X, Y)

        self.assertTrue(len(w) == 2)
        self.assertTrue(np.all(w.keys == X.keys))
        self.assertTrue(
            np.allclose(w.block(0).values, np.array([-1.0, 1.0]), rtol=1e-13)
        )
        self.assertTrue(
            np.allclose(w.block(1).values, np.array([-2.0, 2.0]), rtol=1e-7)
        )
        for key, blockw in w:
            self.assertTrue(np.all(blockw.samples == Y.block(key).properties))
            self.assertTrue(np.all(blockw.properties == X.block(key).properties))

        Ydot = fn.dot(X, w)
        self.assertTrue(np.all(Ydot.keys == Y.keys))
        for key, expected_block in Ydot:
            comparing_dict = compare_blocks(expected_block, Y.block(key), rtol=1e-3)
            if not comparing_dict["general"]:
                print(str(comparing_dict))
            self.assertTrue(comparing_dict["general"])
            self.assertTrue(
                np.allclose(expected_block.values, Y.block(key).values, rtol=1e-13)
            )


# TODO: add tests with torch & torch scripting/tracing

if __name__ == "__main__":
    unittest.main()
