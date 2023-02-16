import os
import unittest

import numpy as np

import equistore.io
import equistore.operations as fn
from equistore import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
TEST_FILE = "qm7-spherical-expansion.npz"


class TestUniqueMetadata(unittest.TestCase):
    """Finding unique_metadata metadata along a given axis of a TensorMap or
    TensorBlock"""

    def setUp(self):
        self.tensor1 = test_tensor_map()  # Test case 1
        self.tensor2 = test_large_tensor_map()  # Test case 2
        self.tensor3 = equistore.io.load(  # Test case 3
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )

    def test_unique_metadata_block_samples(self):
        """Tests getting unique metadata along properties axis w/ 3 test cases"""
        # Test case 1
        samples = [0, 1, 2, 5]
        target_samples = Labels(
            names=["samples"], values=np.array([[s] for s in samples])
        )
        actual_samples = fn.unique_metadata_block(
            self.tensor1.block(3), "samples", "samples"
        )
        self.assertTrue(
            fn._utils._labels_equal(target_samples, actual_samples, exact_order=True)
        )
        # Test case 2
        samples = [0, 2, 4]
        target_samples = Labels(
            names=["samples"], values=np.array([[s] for s in samples])
        )
        actual_samples = fn.unique_metadata_block(
            self.tensor2.block(0), "samples", "samples"
        )
        self.assertTrue(
            fn._utils._labels_equal(target_samples, actual_samples, exact_order=True)
        )
        # Test case 3
        samples = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        target_samples = Labels(
            names=["structure"], values=np.array([[s] for s in samples])
        )
        actual_samples = fn.unique_metadata_block(
            self.tensor3.block(0), "samples", "structure"
        )
        self.assertTrue(
            fn._utils._labels_equal(target_samples, actual_samples, exact_order=True)
        )

    def test_unique_metadata_block_samples_to_zero(self):
        """Tests getting unique samples metadata of a block sliced to zero
        dimension"""
        # Slice block to zero along samples and check labels return is empty
        # with correct names
        target_samples = Labels(names=["structure"], values=np.array([[0]])[:0])
        sliced_block = fn.slice_block(
            self.tensor3.block(0),
            samples=Labels(names=["structure"], values=np.array([[-1]])),
        )
        actual_samples = fn.unique_metadata_block(sliced_block, "samples", "structure")
        self.assertTrue(
            fn._utils._labels_equal(target_samples, actual_samples, exact_order=True)
        )
        self.assertTrue(len(actual_samples) == 0)

    def test_unique_metadata_block_properties(self):
        """Tests getting unique metadata along properties axis w/ 3 test
        cases"""
        properties = [0]
        target_properties = Labels(
            names=["properties"], values=np.array([[p] for p in properties])
        )
        actual_properties = fn.unique_metadata_block(
            self.tensor1.block(3), "properties", "properties"
        )
        self.assertTrue(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )
        # Test case 2
        properties = [3, 4, 5]
        target_properties = Labels(
            names=["properties"], values=np.array([[p] for p in properties])
        )
        actual_properties = fn.unique_metadata_block(
            self.tensor2.block(1), "properties", "properties"
        )
        self.assertTrue(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )
        # Test case 3
        properties = [0, 1, 2, 3]
        target_properties = Labels(
            names=["n"], values=np.array([[p] for p in properties])
        )
        actual_properties = fn.unique_metadata_block(
            self.tensor3.block(0), "properties", "n"
        )
        self.assertTrue(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )

    def test_unique_metadata_block_properties_to_zero(self):
        """Tests getting unique properties metadata of a block sliced to zero
        dimension"""
        # Slice block to zero along properties and check labels return is empty
        # with correct names
        target_properties = Labels(names=["n"], values=np.array([[0]])[:0])
        sliced_block = fn.slice_block(
            self.tensor3.block(0),
            properties=Labels(names=["n"], values=np.array([[-1]])),
        )
        actual_properties = fn.unique_metadata_block(sliced_block, "properties", "n")
        self.assertTrue(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )
        self.assertTrue(len(actual_properties) == 0)

    def test_unique_metadata_block_gradient_samples(self):
        # Test case 1
        parameter = "parameter"
        samples = [0, 2]
        target_samples = Labels(
            names=["sample"], values=np.array([[s] for s in samples])
        )
        actual_samples = fn.unique_metadata_block(
            self.tensor1.block(0), "samples", "sample", parameter
        )
        self.assertTrue(
            fn._utils._labels_equal(target_samples, actual_samples, exact_order=True)
        )
        # Test case 2
        parameter = "parameter"
        samples, sname = [[0, -2], [0, 3], [2, -2]], ["sample", "parameter"]
        target_samples = Labels(names=sname, values=np.array([s for s in samples]))
        actual_samples = fn.unique_metadata_block(
            self.tensor2.block(1), "samples", sname, parameter
        )
        self.assertTrue(
            fn._utils._labels_equal(target_samples, actual_samples, exact_order=True)
        )
        # Test case 3
        target_samples = Labels(
            names=["sample"], values=np.arange(0, 52).reshape(-1, 1)
        )
        actual_samples = fn.unique_metadata_block(
            self.tensor3.block(1), "samples", "sample", "cell"
        )
        self.assertTrue(
            fn._utils._labels_equal(target_samples, actual_samples, exact_order=True)
        )

    def test_unique_metadata_block_gradient_samples_to_zero(self):
        """Test unique samples metadata of block sliced to zero"""
        # Slice block to zero along samples and check labels return is empty
        # with correct names
        target_samples = Labels(names=["sample"], values=np.array([[0]])[:0])
        sliced_block = fn.slice_block(
            self.tensor3.block(0),
            samples=Labels(names=["structure"], values=np.array([[-1]])),
        )
        actual_samples = fn.unique_metadata_block(
            sliced_block, "samples", "sample", gradient_param="cell"
        )
        self.assertTrue(
            fn._utils._labels_equal(target_samples, actual_samples, exact_order=True)
        )
        self.assertTrue(len(actual_samples) == 0)

    def test_unique_metadata_block_gradient_properties(self):
        """Test unique properties metadata of gradient blocks"""
        # Test case 1
        parameter = "parameter"
        properties = [0]
        target_properties = Labels(
            names=["properties"], values=np.array([[p] for p in properties])
        )
        actual_properties = fn.unique_metadata_block(
            self.tensor1.block(0), "properties", "properties", parameter
        )
        self.assertTrue(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )
        # Test case 2
        parameter = "parameter"
        properties, pname = [[3], [4], [5]], ["properties"]
        target_properties = Labels(
            names=pname, values=np.array([p for p in properties])
        )
        actual_properties = fn.unique_metadata_block(
            self.tensor2.block(1), "properties", pname, parameter
        )
        self.assertTrue(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )
        # By definition the unique TensorBlock and Gradient block properties
        # should be equivalent
        actual_properties_tensorblock = fn.unique_metadata_block(
            self.tensor2.block(1),
            "properties",
            pname,
        )
        self.assertTrue(
            fn._utils._labels_equal(
                actual_properties_tensorblock, actual_properties, exact_order=True
            )
        )
        # Test case 3
        target_properties = Labels(names=["n"], values=np.arange(0, 4).reshape(-1, 1))
        actual_properties = fn.unique_metadata_block(
            self.tensor3.block(1), "properties", "n", "positions"
        )
        self.assertTrue(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )
        # Passing names as list
        actual_properties = fn.unique_metadata_block(
            self.tensor3.block(1), "properties", ["n"], "positions"
        )
        self.assertTrue(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )
        # Passing names as tuple
        actual_properties = fn.unique_metadata_block(
            self.tensor3.block(1), "properties", ("n",), "positions"
        )
        self.assertTrue(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )

    def test_unique_metadata_block_gradient_properties_to_zero(self):
        """Test unique samples metadata of block sliced to zero"""
        # Slice block to zero along properties and check labels return is empty
        # with correct names
        target_properties = Labels(names=["n"], values=np.array([[0]])[:0])
        sliced_block = fn.slice_block(
            self.tensor3.block(0),
            properties=Labels(names=["n"], values=np.array([[-1]])),
        )
        actual_properties = fn.unique_metadata_block(
            sliced_block, "properties", "n", gradient_param="cell"
        )
        self.assertTrue(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )
        self.assertTrue(len(actual_properties) == 0)

    def test_unique_indices_samples(self):
        """Test unique on whole TensorMap along samples"""
        # Test case 1
        samples = [0, 1, 2, 3, 4, 5, 6, 8]
        target_samples = Labels(
            names=["samples"], values=np.array([[s] for s in samples])
        )
        actual_samples = fn.unique_metadata(self.tensor1, "samples", "samples")
        self.assertTrue(
            fn._utils._labels_equal(target_samples, actual_samples, exact_order=True)
        )
        # Test case 2
        actual_samples = fn.unique_metadata(self.tensor2, "samples", "samples")
        self.assertTrue(
            fn._utils._labels_equal(actual_samples, target_samples, exact_order=True)
        )

    def test_unique_indices_samples_different_types(self):
        """Passign names as different types"""
        samples = [0, 1, 2, 3, 4, 5, 6, 8]
        target_samples = Labels(
            names=["samples"], values=np.array([[s] for s in samples])
        )
        # Passing names as list
        actual_samples = fn.unique_metadata(self.tensor2, "samples", ["samples"])
        self.assertTrue(
            fn._utils._labels_equal(target_samples, actual_samples, exact_order=True)
        )
        # Passing names as tuple
        actual_samples = fn.unique_metadata(self.tensor2, "samples", ("samples",))
        self.assertTrue(
            fn._utils._labels_equal(target_samples, actual_samples, exact_order=True)
        )

    def test_unique_indices_properties(self):
        """Test unique on whole TensorMap along samples"""
        # Test case 1
        properties = [0, 3, 4, 5]
        target_properties = Labels(
            names=["properties"], values=np.array([[p] for p in properties])
        )
        actual_properties = fn.unique_metadata(self.tensor1, "properties", "properties")
        self.assertTrue(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )
        # Test case 2
        properties = [0, 1, 2, 3, 4, 5, 6, 7]
        target_properties = Labels(
            names=["properties"], values=np.array([[p] for p in properties])
        )
        actual_properties = fn.unique_metadata(self.tensor2, "properties", "properties")
        self.assertTrue(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )

    def test_unique_indices_properties_different_types(self):
        """Passing names as different types"""
        properties = [0, 1, 2, 3, 4, 5, 6, 7]
        target_properties = Labels(
            names=["properties"], values=np.array([[p] for p in properties])
        )
        # Passing names as list
        actual_properties = fn.unique_metadata(
            self.tensor2, "properties", ["properties"]
        )
        self.assertTrue(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )
        # Passing names as tuple
        actual_properties = fn.unique_metadata(
            self.tensor2, "properties", ("properties",)
        )
        self.assertTrue(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )

    def test_unique_indices_gradients_samples(self):
        """Tests getting unique metadata of gradients on whole TensorMap along
        samples"""
        parameter = "parameter"
        samples = np.array([[0, -2], [0, 1], [0, 3], [1, -2], [2, -2], [2, 3], [3, 3]])
        snames = ["sample", "parameter"]
        target_samples = Labels(names=snames, values=samples)
        actual_samples = fn.unique_metadata(self.tensor1, "samples", snames, parameter)
        self.assertTrue(
            fn._utils._labels_equal(actual_samples, target_samples, exact_order=True)
        )

    def test_unique_indices_gradients_samples_incorrect_order(self):
        """
        Tests false of getting unique metadata of gradients on whole TensorMap
        along samples with incorrect order
        """
        parameter = "parameter"
        # Incorrect order samples - [1, -2] and [2, -2] switched around
        samples = np.array([[0, -2], [0, 1], [0, 3], [2, -2], [1, -2], [2, 3], [3, 3]])
        snames = ["sample", "parameter"]
        target_samples = Labels(names=snames, values=samples)
        actual_samples = fn.unique_metadata(self.tensor1, "samples", snames, parameter)
        self.assertFalse(
            fn._utils._labels_equal(actual_samples, target_samples, exact_order=True)
        )

    def test_unique_indices_gradients_properties(self):
        """Tests getting unique metadata of gradients on whole TensorMap along
        properties"""
        parameter = "parameter"
        properties, pnames = [0, 3, 4, 5], ["properties"]
        target_properties = Labels(
            names=pnames, values=np.array([[p] for p in properties])
        )
        actual_properties = fn.unique_metadata(
            self.tensor1, "properties", pnames, parameter
        )
        self.assertTrue(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )

    def test_unique_indices_gradients_properties_incorrect_order(self):
        """
        Tests false of getting unique metadata of gradients on whole TensorMap
        along properties with incorrect order
        """
        # 3 and 4 switched
        parameter = "parameter"
        properties, pnames = [0, 4, 3, 5], ["properties"]
        target_properties = Labels(
            names=pnames, values=np.array([[p] for p in properties])
        )
        actual_properties = fn.unique_metadata(
            self.tensor1, "properties", pnames, parameter
        )
        self.assertFalse(
            fn._utils._labels_equal(
                target_properties, actual_properties, exact_order=True
            )
        )


class TestUniqueMetadataErrors(unittest.TestCase):
    """
    Testing the errors raised by :py:func:`unique_metadata` and
    :py:func:`unique_metadata_block`
    """

    def setUp(self):
        self.tensor = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        self.block = self.tensor.block(0)

    def test_unique_indices_block_type_errors(self):
        # TypeError with TM
        with self.assertRaises(TypeError) as cm:
            fn.unique_metadata_block(self.tensor, "samples", ["structure"])
        self.assertEqual(
            str(cm.exception),
            "`block` must be an equistore `TensorBlock`",
        )
        # TypeError axis as float
        with self.assertRaises(TypeError) as cm:
            fn.unique_metadata_block(self.block, 3.14, ["structure"])
        self.assertEqual(
            str(cm.exception),
            "`axis` must be a `str`, either `'samples'` or `'properties'`",
        )
        # TypeError names as float
        with self.assertRaises(TypeError) as cm:
            fn.unique_metadata_block(self.block, "properties", 3.14)
        self.assertEqual(str(cm.exception), "`names` must be a `list` of `str`")
        # TypeError gradient_param as float
        with self.assertRaises(TypeError) as cm:
            fn.unique_metadata_block(
                self.block, "properties", ["structure"], gradient_param=3.14
            )
        self.assertEqual(str(cm.exception), "`gradient_param` must be a `str`")

    def test_unique_indices_block_value_errors(self):
        """tests raising of value errors"""
        # ValueError axis not "samples" or "properties"
        with self.assertRaises(ValueError) as cm:
            fn.unique_metadata_block(self.block, "ciao", ["structure"])
        self.assertEqual(
            str(cm.exception),
            "`axis` must be passsed as either `'samples'` or `'properties'`",
        )
        # ValueError names not in block
        with self.assertRaises(ValueError) as cm:
            fn.unique_metadata_block(self.block, "properties", ["ciao"])
        self.assertEqual(
            str(cm.exception),
            "the block(s) passed must have samples/properties"
            + " names that matches the one passed in `names`",
        )
        # ValueError gradient_param not a valid param for gradients
        with self.assertRaises(TypeError) as cm:
            fn.unique_metadata_block(
                self.block, "properties", ["structure"], gradient_param=3.14
            )
        self.assertEqual(str(cm.exception), "`gradient_param` must be a `str`")

    def test_unique_indices_block_no_errors(self):
        """tests no raising of errors"""
        # No error names as str, list of str, or tuple of str
        fn.unique_metadata_block(self.block, "samples", "structure")
        fn.unique_metadata_block(self.block, "samples", ["structure"])
        fn.unique_metadata_block(self.block, "samples", ("structure",))

    def test_unique_type_errors(self):
        """tests raising of type errors"""
        # TypeError with TB
        with self.assertRaises(TypeError) as cm:
            fn.unique_metadata(self.block, "samples", ["structure"])
        self.assertEqual(
            str(cm.exception),
            "`tensor` must be an equistore `TensorMap`",
        )
        # TypeError axis as float
        with self.assertRaises(TypeError) as cm:
            fn.unique_metadata(self.tensor, 3.14, ["structure"])
        self.assertEqual(
            str(cm.exception),
            "`axis` must be a `str`, either `'samples'` or `'properties'`",
        )
        # TypeError names as float
        with self.assertRaises(TypeError) as cm:
            fn.unique_metadata(self.tensor, "properties", 3.14)
        self.assertEqual(str(cm.exception), "`names` must be a `list` of `str`")

    def test_unique_value_errors(self):
        """tests raising of value errors"""
        # ValueError axis not "samples" or "properties"
        with self.assertRaises(ValueError) as cm:
            fn.unique_metadata(self.tensor, "ciao", ["structure"])
        self.assertEqual(
            str(cm.exception),
            "`axis` must be passsed as either `'samples'` or `'properties'`",
        )
        # ValueError names not in block
        with self.assertRaises(ValueError) as cm:
            fn.unique_metadata(self.tensor, "properties", ["ciao"])
        self.assertEqual(
            str(cm.exception),
            "the block(s) passed must have samples/properties"
            + " names that matches the one passed in `names`",
        )

    def test_unique_no_errors(self):
        """tests no raising of errors"""
        # No error names as str, list of str, or tuple of str
        fn.unique_metadata(self.tensor, "properties", "n")
        fn.unique_metadata(self.tensor, "properties", ["n"])
        fn.unique_metadata(self.tensor, "properties", ("n",))


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
