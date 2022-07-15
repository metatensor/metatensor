import unittest

import numpy as np
from utils import test_tensor_map

from equistore import Labels


class TestLabels(unittest.TestCase):
    def test_python_labels(self):
        labels = Labels(
            names=["a", "b"],
            values=np.array([[0, 0]], dtype=np.int32),
        )

        self.assertEqual(labels.names, ("a", "b"))
        self.assertEqual(len(labels), 1)
        self.assertEqual(tuple(labels[0]), (0, 0))

        self.assertTrue(np.all(labels.asarray() == np.array([[0, 0]])))

    def test_native_labels(self):
        tensor = test_tensor_map()
        labels = tensor.keys

        self.assertEqual(labels.names, ("key_1", "key_2"))
        self.assertEqual(len(labels), 4)
        self.assertEqual(tuple(labels[0]), (0, 0))
        self.assertEqual(tuple(labels[1]), (1, 0))
        self.assertEqual(tuple(labels[2]), (2, 2))
        self.assertEqual(tuple(labels[3]), (2, 3))

        expected = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 2],
                [2, 3],
            ]
        )
        self.assertTrue(np.all(labels.asarray() == expected))

    def test_position(self):
        tensor = test_tensor_map()
        labels = tensor.keys

        self.assertEqual(labels.position((0, 0)), 0)
        self.assertEqual(labels.position((2, 3)), 3)
        self.assertEqual(labels.position((2, -1)), None)

    def test_contains(self):
        tensor = test_tensor_map()
        labels = tensor.keys

        self.assertTrue((0, 0) in labels)
        self.assertTrue((2, 3) in labels)
        self.assertFalse((2, -1) in labels)

    def test_named_tuples(self):
        tensor = test_tensor_map()
        labels = tensor.keys

        iterator = labels.as_namedtuples()
        first = next(iterator)

        self.assertTrue(isinstance(first, tuple))
        self.assertTrue(hasattr(first, "_fields"))

        self.assertEqual(first.key_1, 0)
        self.assertEqual(first.key_2, 0)
        self.assertEqual(first.as_dict(), {"key_1": 0, "key_2": 0})

        self.assertEqual(first, (0, 0))
        self.assertEqual(next(iterator), (1, 0))
        self.assertEqual(next(iterator), (2, 2))
        self.assertEqual(next(iterator), (2, 3))

        self.assertRaises(StopIteration, next, iterator)

    def test_not_writeable(self):
        tensor = test_tensor_map()
        labels = tensor.keys

        with self.assertRaises(ValueError) as cm:
            labels[0][0] = 4

        self.assertEqual(str(cm.exception), "assignment destination is read-only")


if __name__ == "__main__":
    unittest.main()
