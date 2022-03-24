import unittest
import numpy as np

from aml_storage import Labels
from utils import test_descriptor


class TestLabels(unittest.TestCase):
    def test_python_labels(self):
        labels = Labels(
            names=["a", "b"],
            values=np.array([[0, 0]], dtype=np.int32),
        )

        self.assertEqual(labels.names, ("a", "b"))
        self.assertEqual(len(labels), 1)
        self.assertEqual(tuple(labels[0]), (0, 0))

    def test_native_labels(self):
        descriptor = test_descriptor()
        labels = descriptor.sparse

        self.assertEqual(labels.names, ("sparse_1", "sparse_2"))
        self.assertEqual(len(labels), 4)
        self.assertEqual(tuple(labels[0]), (0, 0))
        self.assertEqual(tuple(labels[1]), (1, 0))
        self.assertEqual(tuple(labels[2]), (2, 2))
        self.assertEqual(tuple(labels[3]), (2, 3))

    def test_position(self):
        descriptor = test_descriptor()
        labels = descriptor.sparse

        self.assertEqual(labels.position((0, 0)), 0)
        self.assertEqual(labels.position((2, 3)), 3)
        self.assertEqual(labels.position((2, -1)), None)

    def test_contains(self):
        descriptor = test_descriptor()
        labels = descriptor.sparse

        self.assertTrue((0, 0) in labels)
        self.assertTrue((2, 3) in labels)
        self.assertFalse((2, -1) in labels)

    def test_named_tuples(self):
        descriptor = test_descriptor()
        labels = descriptor.sparse

        iterator = labels.as_namedtuples()
        first = next(iterator)

        self.assertTrue(isinstance(first, tuple))
        self.assertTrue(hasattr(first, "_fields"))

        self.assertEqual(first.sparse_1, 0)
        self.assertEqual(first.sparse_2, 0)
        self.assertEqual(first.as_dict(), {"sparse_1": 0, "sparse_2": 0})

        self.assertEqual(first, (0, 0))
        self.assertEqual(next(iterator), (1, 0))
        self.assertEqual(next(iterator), (2, 2))
        self.assertEqual(next(iterator), (2, 3))

        self.assertRaises(StopIteration, next, iterator)


if __name__ == "__main__":
    unittest.main()
