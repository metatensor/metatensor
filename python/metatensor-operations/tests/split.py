import os
import unittest
from typing import List, Union

import numpy as np

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
TEST_FILE_1 = "qm7-spherical-expansion.npz"
TEST_FILE_2 = "qm7-power-spectrum.npz"


def _equivalent_labels(a: Labels, b: Labels):
    """
    Check that two labels are equivalent, i.e. contain the same values, potentially in a
    different order.
    """

    if a.names != b.names:
        return False

    if len(a) != len(b):
        return False

    for entry in a:
        if entry not in b:
            return False

    return True


class TestSplitSamples(unittest.TestCase):
    """Splitting samples dimension of TensorMap and TensorBlock"""

    def setUp(self):
        self.tensor = metatensor.load(
            # Use the sph. exp. TensorMap - it has components
            os.path.join(DATA_ROOT, TEST_FILE_1),
            use_numpy=True,
        )

    # === TensorBlock checks

    def _check_split_blocks(
        self,
        block: TensorBlock,
        split_blocks: List[TensorBlock],
        grouped_labels: List[Labels],
    ):
        # Same number of returned blocks as groups of indices
        self.assertEqual(len(split_blocks), len(grouped_labels))

        # Define reference values
        target_names = list(grouped_labels[0].names)
        p_size = len(block.properties)
        c_sizes = [len(c) for c in block.components]
        # Checks on each split block
        for i, split_block in enumerate(split_blocks):
            # Check samples indices
            target_idxs = grouped_labels[i]
            actual_idxs = _unique_indices(
                split_block, axis="samples", names=target_names
            )
            self.assertTrue(_equivalent_labels(actual_idxs, target_idxs))
            # No properties split
            self.assertEqual(len(split_block.properties), p_size)
            # No components split
            for c_i, c_size in enumerate(c_sizes):
                self.assertEqual(len(split_block.components[c_i]), c_size)
            # Equal values tensor
            samples_filter = np.array(
                [s in grouped_labels[i] for s in block.samples.view(target_names)]
            )
            self.assertTrue(
                np.all(split_block.values == block.values[samples_filter, ...])
            )
            # Check gradients
            for parameter, gradient in block.gradients():
                split_gradient = split_block.gradient(parameter)
                # No spilt along properties
                self.assertTrue(
                    np.all(split_gradient.properties == gradient.properties)
                )
                # Samples map to original samples of parent block updated
                self.assertLess(
                    np.max(split_gradient.samples["sample"]),
                    split_block.values.shape[0],
                )

                # other columns in the gradient samples have been sliced correctly

                gradient_samples_sample = gradient.samples["sample"]
                gradient_sample_filter = samples_filter[gradient_samples_sample]
                if len(gradient.samples.names) > 1:
                    expected = gradient.samples.values[gradient_sample_filter, 1:]
                    sliced_gradient_samples = split_gradient.samples.values[:, 1:]
                    self.assertTrue(np.all(sliced_gradient_samples == expected))

                # No splitting of components
                self.assertEqual(
                    len(gradient.components), len(split_gradient.components)
                )
                for sliced_c, c in zip(split_gradient.components, gradient.components):
                    self.assertEqual(sliced_c, c)
                expected = gradient.values[gradient_sample_filter]
                self.assertTrue(np.all(split_gradient.values == expected))

    def _check_empty_block(self, block, split_block):
        # sliced block has no values
        self.assertEqual(len(split_block.values.flatten()), 0)
        # sliced block has dimension zero for samples
        self.assertEqual(split_block.values.shape[0], 0)
        # sliced block has original dimension for properties
        self.assertEqual(split_block.values.shape[-1], block.values.shape[-1])
        for parameter, gradient in block.gradients():
            sliced_gradient = split_block.gradient(parameter)
            # no slicing of properties has occurred
            self.assertTrue(np.all(sliced_gradient.properties == gradient.properties))
            # sliced block contains zero samples
            self.assertEqual(sliced_gradient.values.shape[0], 0)

    # === TensorMap checks

    def _check_num_blocks_tensor(self, split_tensors: List[TensorMap]):
        # All returned TensorMaps should have the same number of blocks as the
        # original
        for tensor in split_tensors:
            self.assertEqual(len(tensor.keys), len(self.tensor.keys))

    def _check_group_sizes_tensors(
        self,
        split_tensors: List[TensorMap],
        grouped_labels: List[Labels],
    ):
        # Same number of returned tensors as groups of indices
        self.assertEqual(len(split_tensors), len(grouped_labels))
        # Number of unique idxs in each tensor equal to respective group size
        for i, tensor in enumerate(split_tensors):
            unique_idxs = _unique_indices(
                tensor, axis="samples", names=grouped_labels[i].names
            )
            self.assertEqual(len(unique_idxs), len(grouped_labels[i]))

    def _check_empty_tensor(self, tensor, split_tensor):
        for key in tensor.keys:
            self._check_empty_block(tensor[key], split_tensor[key])

    # === main unit tests

    def test_split_block(self):
        # All indices present - block with key (2, 6, 6)
        block = self.tensor.block(
            spherical_harmonics_l=2, species_center=6, species_neighbor=6
        )  # has structure samples 0 -> 9 (inc.)
        grouped_labels = [
            Labels(names=["structure"], values=np.array([[0], [6], [7]])),
            Labels(names=["structure"], values=np.array([[2], [3], [4]])),
            Labels(names=["structure"], values=np.array([[1], [5], [8], [9]])),
        ]
        split_blocks = metatensor.split_block(
            block, axis="samples", grouped_labels=grouped_labels
        )
        self.assertEqual(
            np.sum([len(b.samples) for b in split_blocks]), len(block.samples)
        )
        self._check_split_blocks(block, split_blocks, grouped_labels)
        # All indices present - block with key (2, 6, 8)
        block = self.tensor.block(
            spherical_harmonics_l=2, species_center=6, species_neighbor=8
        )  # has structure samples 4 -> 6 (inc.)
        grouped_labels = [
            Labels(names=["structure"], values=np.array([[4], [6]])),
            Labels(names=["structure"], values=np.array([[5]])),
        ]
        split_blocks = metatensor.split_block(
            block, axis="samples", grouped_labels=grouped_labels
        )
        self.assertEqual(
            np.sum([len(b.samples) for b in split_blocks]), len(block.samples)
        )
        self._check_split_blocks(block, split_blocks, grouped_labels)
        # Indices not present for first group
        grouped_labels_empty = [
            Labels(names=["structure"], values=np.array([[1], [2]])),  # not present
            Labels(names=["structure"], values=np.array([[4], [6]])),  # present
            Labels(names=["structure"], values=np.array([[5]])),  # present
        ]
        split_blocks = metatensor.split_block(
            block, axis="samples", grouped_labels=grouped_labels_empty
        )
        self._check_split_blocks(block, split_blocks[1:], grouped_labels_empty[1:])
        self._check_empty_block(block, split_blocks[0])

    def test_split(self):
        # Normal - all indices present
        grouped_labels = [
            Labels(names=["structure"], values=np.array([[0], [6], [7]])),
            Labels(names=["structure"], values=np.array([[2], [3], [4]])),
            Labels(names=["structure"], values=np.array([[1], [5], [8], [9]])),
        ]
        split_tensors = metatensor.split(
            self.tensor,
            axis="samples",
            grouped_labels=grouped_labels,
        )
        self._check_num_blocks_tensor(split_tensors)
        self._check_group_sizes_tensors(split_tensors, grouped_labels)
        # Third returned tensor should be empty
        grouped_labels = [
            Labels(names=["structure"], values=np.array([[6], [7]])),  # present
            Labels(names=["structure"], values=np.array([[2], [3], [4]])),  # present
            Labels(
                names=["structure"], values=np.array([[1], [5], [8], [9]]) * -1
            ),  # not present
        ]
        split_tensors = metatensor.split(
            self.tensor,
            axis="samples",
            grouped_labels=grouped_labels,
        )
        self._check_num_blocks_tensor(split_tensors)
        self._check_group_sizes_tensors(split_tensors[:2], grouped_labels[:2])
        self._check_empty_tensor(self.tensor, split_tensors[2])

    def test_split_block_(self):
        # All indices present - block with key (2, 6, 6)
        block = self.tensor.block(
            spherical_harmonics_l=2, species_center=6, species_neighbor=6
        )  # has structure samples 0 -> 9 (inc.)
        grouped_labels = [
            Labels(names=["structure"], values=np.array([[0], [6], [7]])),
            Labels(names=["structure"], values=np.array([[2], [6], [4]])),
            Labels(names=["structure"], values=np.array([[1], [0], [6], [4]])),
        ]
        split_blocks = metatensor.split_block(
            block, axis="samples", grouped_labels=grouped_labels
        )
        self._check_split_blocks(block, split_blocks, grouped_labels)

    def test_no_splitting(self):
        # Passing no groups of indices returns an empty list
        # Block
        self.assertEqual(
            metatensor.split_block(
                self.tensor.block(0), axis="samples", grouped_labels=[]
            ),
            [],
        )
        # TensorMap
        self.assertEqual(
            metatensor.split(self.tensor, axis="samples", grouped_labels=[]), []
        )


class TestSplitProperties(unittest.TestCase):
    """Splitting property dimension of TensorMap and TensorBlock"""

    def setUp(self):
        self.tensor = metatensor.load(
            # Use the pow. spectrum TensorMap - it has no components but
            # mutliple properties
            os.path.join(DATA_ROOT, TEST_FILE_2),
            use_numpy=True,
        )

    # === TensorBlock checks

    def _check_split_blocks(
        self,
        block: TensorBlock,
        split_blocks: List[TensorBlock],
        grouped_labels: List[Labels],
    ):
        # Same number of returned blocks as groups of indices
        self.assertEqual(len(split_blocks), len(grouped_labels))

        # Define reference values
        target_names = list(grouped_labels[0].names)
        s_size = len(block.samples)
        c_sizes = [len(c) for c in block.components]
        # Checks on each split block
        for i, split_block in enumerate(split_blocks):
            # Check properties indices
            target_idxs = grouped_labels[i]
            actual_idxs = _unique_indices(
                split_block, axis="properties", names=target_names
            )
            self.assertTrue(_equivalent_labels(actual_idxs, target_idxs))
            # No samples split
            self.assertEqual(len(split_block.samples), s_size)
            # No components split
            for c_i, c_size in enumerate(c_sizes):
                self.assertEqual(len(split_block.components[c_i]), c_size)
            # Equal values tensor
            properties_filter = np.array(
                [p in grouped_labels[i] for p in block.properties.view(target_names)]
            )
            self.assertTrue(
                np.all(split_block.values == block.values[..., properties_filter])
            )
            # Check gradients
            for parameter, gradient in block.gradients():
                split_gradient = split_block.gradient(parameter)
                # No splitting of samples
                self.assertTrue(np.all(split_gradient.samples == gradient.samples))
                # No splitting of components
                self.assertEqual(
                    len(gradient.components), len(split_gradient.components)
                )
                for sliced_c, c in zip(split_gradient.components, gradient.components):
                    self.assertTrue(np.all(sliced_c == c))
                # Properties sliced correctly
                self.assertEqual(
                    len(split_gradient.properties),
                    len(
                        [
                            p
                            for p in gradient.properties.view(target_names)
                            if p in target_idxs
                        ]
                    ),
                )
                # Correct gradient values
                self.assertTrue(
                    np.all(
                        split_gradient.values == gradient.values[..., properties_filter]
                    )
                )

    def _check_empty_block(self, block, split_block):
        # sliced block has no values
        self.assertEqual(len(split_block.values.flatten()), 0)
        # sliced block has dimension zero for properties
        self.assertEqual(split_block.values.shape[-1], 0)
        # sliced block has original dimension for samples
        self.assertEqual(split_block.values.shape[0], block.values.shape[0])

        for parameter, gradient in block.gradients():
            split_gradient = split_block.gradient(parameter)
            # no slicing of samples has occurred
            self.assertTrue(np.all(split_gradient.samples == gradient.samples))

            # sliced block contains zero properties
            self.assertEqual(split_gradient.values.shape[-1], 0)

    # === TensorMap checks

    def _check_num_blocks_tensor(self, split_tensors: List[TensorMap]):
        # All returned TensorMaps should have the same number of blocks as the
        # original
        for tensor in split_tensors:
            self.assertEqual(len(tensor.keys), len(self.tensor.keys))

    def _check_group_sizes_tensors(
        self,
        split_tensors: List[TensorMap],
        grouped_labels: List[Labels],
    ):
        # Same number of returned tensors as groups of indices
        self.assertEqual(len(split_tensors), len(grouped_labels))
        # Number of unique idxs in each tensor equal to respective group size
        for i, tensor in enumerate(split_tensors):
            unique_idxs = _unique_indices(
                tensor, axis="properties", names=grouped_labels[i].names
            )
            self.assertEqual(len(unique_idxs), len(grouped_labels[i]))

    def _check_empty_tensor(self, tensor, split_tensor):
        for key in tensor.keys:
            self._check_empty_block(tensor[key], split_tensor[key])

    # === main unit tests

    def test_split_block(self):
        # All indices present - block with key (8, 6, 8) for properties "l" and "n2"
        block = self.tensor.block(
            species_center=8, species_neighbor_1=6, species_neighbor_2=8
        )
        grouped_labels = [
            Labels(names=["l", "n2"], values=np.array([[0, 0], [1, 3], [3, 1]])),
            Labels(names=["l", "n2"], values=np.array([[4, 2], [4, 3], [4, 1]])),
            Labels(names=["l", "n2"], values=np.array([[3, 2], [1, 1]])),
        ]
        split_blocks = metatensor.split_block(
            block, axis="properties", grouped_labels=grouped_labels
        )
        self._check_split_blocks(block, split_blocks, grouped_labels)
        # Indices not present for last group
        grouped_labels_empty = [
            Labels(
                names=["l", "n2"], values=np.array([[0, 0], [1, 3], [3, 1]])
            ),  # present
            Labels(
                names=["l", "n2"], values=np.array([[4, 2], [4, 3], [4, 1]])
            ),  # present
            Labels(
                names=["l", "n2"], values=np.array([[3, 2], [1, 1]]) * -1
            ),  # not present
        ]
        split_blocks = metatensor.split_block(
            block, axis="properties", grouped_labels=grouped_labels_empty
        )
        self._check_split_blocks(block, split_blocks[:2], grouped_labels_empty[:2])
        self._check_empty_block(block, split_blocks[2])

    def test_split(self):
        # Normal - all indices present
        grouped_labels = [
            Labels(names=["l", "n2"], values=np.array([[0, 0], [1, 3], [3, 1]])),
            Labels(names=["l", "n2"], values=np.array([[4, 2], [4, 3], [4, 1]])),
            Labels(names=["l", "n2"], values=np.array([[3, 2], [1, 1]])),
        ]
        split_tensors = metatensor.split(
            self.tensor,
            axis="properties",
            grouped_labels=grouped_labels,
        )
        self._check_num_blocks_tensor(split_tensors)
        self._check_group_sizes_tensors(split_tensors, grouped_labels)
        # Second returned tensor should be empty
        grouped_labels = [
            Labels(
                names=["l", "n2"], values=np.array([[0, 0], [1, 3], [3, 1]])
            ),  # present
            Labels(
                names=["l", "n2"], values=np.array([[4, 2], [4, 3], [4, 1]]) * -1
            ),  # not present
            Labels(names=["l", "n2"], values=np.array([[3, 2], [1, 1]])),  # present
        ]
        split_tensors = metatensor.split(
            self.tensor,
            axis="properties",
            grouped_labels=grouped_labels,
        )
        self._check_num_blocks_tensor(split_tensors)
        self._check_group_sizes_tensors(
            [split_tensors[0], split_tensors[2]], [grouped_labels[0], grouped_labels[2]]
        )
        self._check_empty_tensor(self.tensor, split_tensors[1])

    def test_no_splitting(self):
        # Passing no groups of indices returns an empty list
        # Block
        self.assertEqual(
            metatensor.split_block(
                self.tensor.block(0), axis="properties", grouped_labels=[]
            ),
            [],
        )
        # TensorMap
        self.assertEqual(
            metatensor.split(self.tensor, axis="properties", grouped_labels=[]), []
        )


class TestSplitErrors(unittest.TestCase):
    def setUp(self):
        self.tensor = metatensor.load(
            os.path.join(DATA_ROOT, TEST_FILE_1),
            use_numpy=True,
        )
        self.block = self.tensor.block(0)
        self.grouped_labels = [
            Labels(names=["structure"], values=np.array([[0], [6], [7]])),
            Labels(names=["structure"], values=np.array([[2], [3], [4]])),
            Labels(names=["structure"], values=np.array([[1], [5], [8], [9]])),
        ]

    def test_split_errors(self):
        # TypeError not TM
        with self.assertRaises(TypeError) as cm:
            metatensor.split(
                self.block, axis="samples", grouped_labels=self.grouped_labels
            ),
        self.assertEqual(
            str(cm.exception),
            "`tensor` must be a metatensor TensorMap, "
            "not <class 'metatensor.core.block.TensorBlock'>",
        )

        # axis not str
        with self.assertRaises(TypeError) as cm:
            metatensor.split(
                self.tensor, axis=3.14, grouped_labels=self.grouped_labels
            ),
        self.assertEqual(
            str(cm.exception), "axis must be a string, not <class 'float'>"
        )

        # axis not "samples" or "properties"
        with self.assertRaises(ValueError) as cm:
            metatensor.split(
                self.tensor,
                axis="buongiorno!",
                grouped_labels=self.grouped_labels,
            ),
        self.assertEqual(
            str(cm.exception),
            "axis must be either 'samples' or 'properties'",
        )

        # grouped_labels is Labels not list
        with self.assertRaises(TypeError) as cm:
            metatensor.split(
                self.tensor, axis="samples", grouped_labels=self.grouped_labels[0]
            ),
        self.assertEqual(
            str(cm.exception),
            "`grouped_labels` must be a list, "
            "not <class 'metatensor.core.labels.Labels'>",
        )

        # grouped_labels is list of str
        with self.assertRaises(TypeError) as cm:
            metatensor.split(
                self.tensor, axis="samples", grouped_labels=["a", "b", "c"]
            ),
        self.assertEqual(
            str(cm.exception),
            "`grouped_labels` elements must be metatensor Labels, not <class 'str'>",
        )

        # different names in labels of grouped_labels
        grouped_labels = [
            Labels(names=["red"], values=np.array([[0], [6], [7]])),
            Labels(names=["red"], values=np.array([[2], [3], [4]])),
            Labels(names=["wine"], values=np.array([[1], [5], [8], [9]])),
        ]
        with self.assertRaises(ValueError) as cm:
            metatensor.split(
                self.tensor, axis="samples", grouped_labels=grouped_labels
            ),
        self.assertEqual(
            str(cm.exception),
            "the dimensions names of all Labels in `grouped_labels` must be the same, "
            "got ['red'] and ['wine']",
        )

        # a name in grouped_labels not in the tensor
        grouped_labels = [
            Labels(
                names=["front_and", "center"], values=np.array([[0, 1], [6, 7], [7, 4]])
            ),
            Labels(
                names=["front_and", "center"], values=np.array([[2, 4], [3, 3], [4, 7]])
            ),
            Labels(
                names=["front_and", "center"],
                values=np.array([[1, 5], [5, 3], [8, 10]]),
            ),
        ]
        with self.assertRaises(ValueError) as cm:
            metatensor.split(
                self.tensor,
                axis="samples",
                grouped_labels=grouped_labels,
            )

        self.assertEqual(
            str(cm.exception),
            "the 'front_and' dimension name in `grouped_labels` is not part of "
            "the samples names of the input tensor",
        )

    def test_split_block_errors(self):
        with self.assertRaises(TypeError) as cm:
            metatensor.split_block(
                self.tensor, axis="samples", grouped_labels=self.grouped_labels
            ),

        self.assertEqual(
            str(cm.exception),
            "`block` must be a metatensor TensorBlock, "
            "not <class 'metatensor.core.tensor.TensorMap'>",
        )


def _unique_indices(
    tensor: Union[TensorMap, TensorBlock],
    axis: str,
    names: Union[List[str], str],
):
    """
    For a given ``axis`` (either "s" or "p"), and for the given
    samples/properties ``names``, returns a :py:class:`Labels` object of the
    unique indices in the input ``tensor``, which can be either a
    :py:class:`TensorMap` or :py:class:`TensorBlock` object.
    """
    # Parse tensor into a list of blocks
    if isinstance(tensor, TensorMap):
        blocks = tensor.blocks()
    else:  # TensorBlock
        blocks = [tensor]

    # Parse the names into a list
    names = [names] if isinstance(names, str) else names
    names = list(names) if isinstance(names, tuple) else names

    # Extract indices from each block
    all_idxs = []
    for block in blocks:
        if axis == "samples":
            block_idxs = block.samples.view(names)
        else:
            block_idxs = block.properties.view(names)

        for idx in block_idxs:
            all_idxs.append(idx)
    if len(all_idxs) == 0:  # return an empty Labels, with the correct name
        return Labels(names=names, values=np.array([[i for i in range(len(names))]]))[
            :0
        ]

    # Define the unique indices, convert to a Labels obj, and return
    unique_idxs = np.unique(all_idxs, axis=0)  # this also sorts the idxs
    return Labels(names=names, values=np.array([[j for j in i] for i in unique_idxs]))


if __name__ == "__main__":
    unittest.main()
