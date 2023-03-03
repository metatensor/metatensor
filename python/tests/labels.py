import unittest

import numpy as np
from utils import test_tensor_map

from equistore import EquistoreError, Labels


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

        # check we can use single str for single name
        labels = Labels(
            names=["a"],
            values=np.array([[1]], dtype=np.int32),
        )
        labels_str = Labels(
            names="a",
            values=np.array([[1]], dtype=np.int32),
        )

        self.assertEqual(labels.names, ("a",))
        self.assertEqual(labels.names, labels_str.names)
        self.assertEqual(len(labels), 1)
        self.assertEqual(len(labels_str), 1)
        self.assertEqual(tuple(labels_str[0]), (1,))
        self.assertEqual(tuple(labels_str[0]), tuple(labels[0]))

        self.assertTrue(np.all(labels.asarray() == np.array([[1]])))
        self.assertTrue(np.all(labels.asarray() == labels_str.asarray()))

        # check empty arrays
        labels = Labels(
            names=[],
            values=np.array([[]], dtype=np.int32),
        )

        labels_str = Labels(
            names="",
            values=np.array([[]], dtype=np.int32),
        )

        self.assertEqual(labels.names, [])
        self.assertEqual(labels.names, labels_str.names)
        self.assertEqual(len(labels), 1)
        self.assertEqual(len(labels_str), 1)
        self.assertEqual(tuple(labels_str[0]), tuple(labels[0]))

        self.assertTrue(np.all(labels.asarray() == np.array([[]])))
        self.assertTrue(np.all(labels.asarray() == labels_str.asarray()))

        # check that we can convert from more than strict 2D arrays of int32
        labels = Labels(
            names=["a", "b"],
            values=np.array([[0, 0]]),
        )

        labels = Labels(
            names=["a", "b"],
            values=np.array([[0, 0]], dtype=np.int64),
        )

        with self.assertRaises(TypeError) as cm:
            labels = Labels(
                names=["a", "b"],
                values=np.array([[0, 0]], dtype=np.float64),
            )
        self.assertEqual(
            str(cm.exception),
            "Labels values must be convertible to integers",
        )

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

    def test_invalid_names(self):
        with self.assertRaises(EquistoreError) as cm:
            _ = Labels(
                names=["not an ident"],
                values=np.array([[0]], dtype=np.int32),
            )
        self.assertEqual(
            str(cm.exception),
            "invalid parameter: 'not an ident' is not a valid label name",
        )

    def test_zero_length_label(self):
        label = Labels(["sample", "structure", "atom"], np.array([]))
        self.assertEqual(len(label), 0)

    def test_labels_single(self):
        label = Labels.single()
        self.assertEqual(label.names, ("_",))
        self.assertEqual(label.shape, (1,))

    def test_labels_empty(self):
        names = (
            "foo",
            "bar",
            "baz",
        )
        label = Labels.empty(names)
        self.assertEqual(label.names, names)
        self.assertEqual(len(label), 0)


if __name__ == "__main__":
    unittest.main()
