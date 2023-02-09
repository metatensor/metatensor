import os
import unittest

import numpy as np

import equistore.io
import equistore.metaoperations as fn
from equistore import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
TEST_FILE = "qm7-spherical-expansion.npz"


class TestUnique(unittest.TestCase):
    """Finding unique metadata along a given axis of a TensorMap or
    TensorBlock"""

    def setUp(self):
        # Test case 1
        self.tensor1 = test_tensor_map()
        self.block1 = self.tensor1.block(3)
        # Test case 2
        self.tensor2 = test_large_tensor_map()
        self.block2 = self.tensor2.block(0)
        # Test case 3
        self.tensor3 = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        self.block3 = self.tensor3.block(0)

    def test_unique_block(self):
        # Test case 1
        samples = [0, 1, 2, 5]
        target_idxs = Labels(names=["samples"], values=np.array([[s] for s in samples]))
        actual_idxs = fn.unique_block(self.block1, "samples", "samples")
        self.assertTrue(_labels_equal(actual_idxs, target_idxs, exact_order=True))
        properties = [0]
        target_idxs = Labels(
            names=["properties"], values=np.array([[p] for p in properties])
        )
        actual_idxs = fn.unique_block(self.block1, "properties", "properties")
        self.assertTrue(_labels_equal(actual_idxs, target_idxs, exact_order=True))
        # Test case 2
        samples = [0, 2, 4]
        target_idxs = Labels(names=["samples"], values=np.array([[s] for s in samples]))
        actual_idxs = fn.unique_block(self.block2, "samples", "samples")
        self.assertTrue(_labels_equal(actual_idxs, target_idxs, exact_order=True))
        # Test case 3
        return

    def test_unique(self):
        samples = [0, 1, 2, 3, 4, 5, 6, 8]
        target_idxs = Labels(names=["samples"], values=np.array([[s] for s in samples]))
        # Test case 1
        actual_idxs = fn.unique(self.tensor1, "samples", "samples")
        self.assertTrue(_labels_equal(actual_idxs, target_idxs, exact_order=True))
        # Test case 2
        actual_idxs = fn.unique(self.tensor2, "samples", "samples")
        self.assertTrue(_labels_equal(actual_idxs, target_idxs, exact_order=True))
        # Passing names as str
        # Passing names as list
        # Passing names as tuple
        return


class TestUniqueErrors(unittest.TestCase):
    """
    Testing the errors raised by :py:func:`unique` and
    :py:func:`unique_block`
    """

    def setUp(self):
        self.tensor = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        self.block = self.tensor.block(0)

    def test_unique_block(self):
        # TypeError with TM
        with self.assertRaises(TypeError) as cm:
            fn.unique_block(self.tensor, "samples", ["structure"])
        self.assertEqual(
            str(cm.exception),
            "``block`` must be an equistore TensorBlock",
        )
        # TypeError axis as float
        with self.assertRaises(TypeError) as cm:
            fn.unique_block(self.block, 3.14, ["structure"])
        self.assertEqual(
            str(cm.exception),
            "``axis`` must be a str, either 'samples' or 'properties'",
        )
        # ValueError axis not "samples" or "properties"
        with self.assertRaises(ValueError) as cm:
            fn.unique_block(self.block, "ciao", ["structure"])
        self.assertEqual(
            str(cm.exception),
            "``axis`` must be passsed as either 'samples' or 'properties'",
        )
        # TypeError names as float
        with self.assertRaises(TypeError) as cm:
            fn.unique_block(self.block, "properties", 3.14)
        self.assertEqual(str(cm.exception), "``names`` must be a list of str")
        # No error names as str, list of str, or tuple of str
        fn.unique_block(self.block, "samples", "structure")
        fn.unique_block(self.block, "samples", ["structure"])
        fn.unique_block(self.block, "samples", ("structure",))
        # ValueError names not in block
        with self.assertRaises(ValueError) as cm:
            fn.unique_block(self.block, "properties", ["ciao"])
        self.assertEqual(
            str(cm.exception),
            "the block(s) passed must have samples/properties"
            + " names that matches the one passed in ``names``",
        )

    def test_unique(self):
        # TypeError with TB
        with self.assertRaises(TypeError) as cm:
            fn.unique(self.block, "samples", ["structure"])
        self.assertEqual(
            str(cm.exception),
            "``tensor`` must be an equistore TensorMap",
        )
        # TypeError axis as float
        with self.assertRaises(TypeError) as cm:
            fn.unique(self.tensor, 3.14, ["structure"])
        self.assertEqual(
            str(cm.exception),
            "``axis`` must be a str, either 'samples' or 'properties'",
        )
        # ValueError axis not "samples" or "properties"
        with self.assertRaises(ValueError) as cm:
            fn.unique(self.tensor, "ciao", ["structure"])
        self.assertEqual(
            str(cm.exception),
            "``axis`` must be passsed as either 'samples' or 'properties'",
        )
        # TypeError names as float
        with self.assertRaises(TypeError) as cm:
            fn.unique(self.tensor, "properties", 3.14)
        self.assertEqual(str(cm.exception), "``names`` must be a list of str")
        # No error names as str, list of str, or tuple of str
        fn.unique(self.tensor, "properties", "n")
        fn.unique(self.tensor, "properties", ["n"])
        fn.unique(self.tensor, "properties", ("n",))
        # ValueError names not in block
        with self.assertRaises(ValueError) as cm:
            fn.unique(self.tensor, "properties", ["ciao"])
        self.assertEqual(
            str(cm.exception),
            "the block(s) passed must have samples/properties"
            + " names that matches the one passed in ``names``",
        )


# def _searchable_labels(labels: Labels):
#     """
#     Returns the input Labels object but after being used to construct a
#     TensorBlock, so that look-ups can be performed.
#     """
#     return TensorBlock(
#         values=np.full((len(labels), 1), 0.0),
#         samples=labels,
#         components=[],
#         properties=Labels(["p"], np.array([[0]], dtype=np.int32)),
#     ).samples


def _labels_equal(a: Labels, b: Labels, exact_order: bool):
    """
    For 2 :py:class:`Labels` objects ``a`` and ``b``, returns true if they are
    exactly equivalent in names, values, and elemental positions. Assumes that
    the Labels are already searchable, i.e. they belogn to a parent TensorBlock
    or TensorMap.
    """
    # They can only be equivalent if the same length
    if len(a) != len(b):
        return False
    if exact_order:
        return np.all(np.array(a == b))
    else:
        return np.all([a_i in b for a_i in a])


def test_tensor_map():
    """
    Create a dummy tensor map to be used in tests. This is the same one as the
    tensor map used in `tensor.rs` tests.
    """
    block_1 = TensorBlock(
        values=np.full((3, 1, 1), 1.0),
        samples=Labels(["samples"], np.array([[0], [2], [4]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_1.add_gradient(
        "parameter",
        samples=Labels(
            ["sample", "parameter"], np.array([[0, -2], [2, 3]], dtype=np.int32)
        ),
        data=np.full((2, 1, 1), 11.0),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
    )

    block_2 = TensorBlock(
        values=np.full((3, 1, 3), 2.0),
        samples=Labels(["samples"], np.array([[0], [1], [3]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[3], [4], [5]], dtype=np.int32)),
    )
    block_2.add_gradient(
        "parameter",
        data=np.full((3, 1, 3), 12.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, -2], [0, 3], [2, -2]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
    )

    block_3 = TensorBlock(
        values=np.full((4, 3, 1), 3.0),
        samples=Labels(["samples"], np.array([[0], [3], [6], [8]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_3.add_gradient(
        "parameter",
        data=np.full((1, 3, 1), 13.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[1, -2]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
    )

    block_4 = TensorBlock(
        values=np.full((4, 3, 1), 4.0),
        samples=Labels(["samples"], np.array([[0], [1], [2], [5]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_4.add_gradient(
        "parameter",
        data=np.full((2, 3, 1), 14.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, 1], [3, 3]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
    )

    # TODO: different number of components?

    keys = Labels(
        names=["key_1", "key_2"],
        values=np.array([[0, 0], [1, 0], [2, 2], [2, 3]], dtype=np.int32),
    )

    return TensorMap(keys, [block_1, block_2, block_3, block_4])


def test_large_tensor_map():
    """
    Create a dummy tensor map of 16 blocks to be used in tests. This is the same
    tensor map used in `tensor.rs` tests.
    """
    tensor = test_tensor_map()
    block_list = [block.copy() for _, block in tensor]

    for i in range(8):
        tmp_bl = TensorBlock(
            values=np.full((4, 3, 1), 4.0),
            samples=Labels(["samples"], np.array([[0], [1], [4], [5]], dtype=np.int32)),
            components=[
                Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))
            ],
            properties=Labels(["properties"], np.array([[i]], dtype=np.int32)),
        )
        tmp_bl.add_gradient(
            "parameter",
            data=np.full((2, 3, 1), 14.0),
            samples=Labels(
                ["sample", "parameter"],
                np.array([[0, 1], [3, 3]], dtype=np.int32),
            ),
            components=[
                Labels(
                    ["components"],
                    np.array([[0], [1], [2]], dtype=np.int32),
                )
            ],
        )
        block_list.append(tmp_bl)

    keys = Labels(
        names=["key_1", "key_2"],
        values=np.array(
            [
                [0, 0],
                [1, 0],
                [2, 2],
                [2, 3],
                [0, 4],
                [1, 4],
                [2, 4],
                [3, 4],
                [0, 5],
                [1, 5],
                [2, 5],
                [3, 5],
            ],
            dtype=np.int32,
        ),
    )
    return TensorMap(keys, block_list)


if __name__ == "__main__":
    unittest.main()
