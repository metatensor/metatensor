import os
import unittest

import numpy as np

import equistore.io
import equistore.operations as fn
from equistore import Labels, TensorBlock, TensorMap

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
TEST_FILE = "qm7-spherical-expansion.npz"


class TestSlice(unittest.TestCase):
    """
    Unit tests for the slice functions.
    Tests are organised into several 'Test Blocks' as follows:
        1) Slicing samples dimension of tensors without gradients.
        2) Slicing samples dimension of tensors with gradients.
        3) Slicing properties dimension of tensors without gradients.
        4) Slicing properties dimension of tensors with gradients.
        5) Slicing samples and properties dimensions simultaneously.
        6) Testing the type handling of the user-facing slice function.
        7) Checking issuance of user warnings when empty blocks are
            created.
    """

    # TEST BLOCK 1: SLICING SAMPLES WITHOUT GRADIENTS

    def test_slice_samples_tensorblock_no_gradients(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Remove the gradients to simplify test
        tensormap = fn.remove_gradients(tensormap)
        # Define a single block to test
        tensorblock = tensormap.block(0)
        # Slice only 'structures' 2, 4, 6, 8
        structures_to_keep = np.arange(2, 10, 2).reshape(-1, 1)
        samples_to_slice = Labels(
            names=["structure"],
            values=structures_to_keep,
        )
        sliced_tensorblock = fn.slice_block(
            tensorblock,
            samples_to_slice=samples_to_slice,
        )
        # Check 1: no slicing of properties has occurred
        self.assertTrue(
            tensorblock.properties.asarray().shape,
            sliced_tensorblock.properties.asarray().shape,
        )
        # Check 2: samples have been sliced to the correct dimension
        self.assertEqual(
            len(sliced_tensorblock.samples),
            len(
                [
                    struct_i
                    for struct_i in tensorblock.samples["structure"]
                    if struct_i in structures_to_keep
                ]
            ),
        )
        # Check 3: samples in sliced block only feature desired strutcure indices
        self.assertTrue(
            np.all(
                [
                    struct_i in structures_to_keep
                    for struct_i in sliced_tensorblock.samples["structure"]
                ]
            )
        )
        # Check 4: no components have been sliced
        for i, comp in enumerate(sliced_tensorblock.components):
            self.assertEqual(len(comp), len(tensorblock.components[i]))

    def test_slice_samples_tensormap_no_gradients(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Remove the gradients to simplify test
        tensormap = fn.remove_gradients(tensormap)
        # Slice only 'structures' 2, 4, 6, 8
        structures_to_keep = np.arange(2, 10, 2).reshape(-1, 1)
        samples_to_slice = Labels(
            names=["structure"],
            values=structures_to_keep,
        )
        sliced_tensormap = fn.slice(
            tensormap,
            samples_to_slice=samples_to_slice,
        )
        for i, (key, block) in enumerate(tensormap):
            # Check 1: no slicing of properties has occurred
            self.assertEqual(
                sliced_tensormap.block(i).properties.asarray().shape,
                block.properties.asarray().shape,
            )
            # Check 2: samples have been sliced to the correct dimension
            self.assertEqual(
                len(sliced_tensormap.block(i).samples),
                len(
                    [
                        struct_i
                        for struct_i in block.samples["structure"]
                        if struct_i in structures_to_keep
                    ]
                ),
            )
            # Check 3: samples in sliced block only feature desired structure indices
            self.assertTrue(
                np.all(
                    [
                        struct_i in structures_to_keep
                        for struct_i in sliced_tensormap.block(i).samples["structure"]
                    ]
                )
            )
            # Check 4: no components have been sliced
            for j, comp in enumerate(block.components):
                self.assertEqual(
                    len(comp), len(sliced_tensormap.block(i).components[j])
                )

        # Check 5: all the keys in the sliced tensormap are in the original
        self.assertTrue(
            np.all([key in tensormap.keys for key in sliced_tensormap.keys])
        )

    def test_slice_samples_tensorblock_empty_no_gradients(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Remove the gradients to simplify test
        tensormap = fn.remove_gradients(tensormap)
        # Define a single block to test
        tensorblock = tensormap.block(0)
        # Slice only 'structures' -1 (i.e. a sample that doesn't exist in the data)
        structures_to_keep = np.array([-1]).reshape(-1, 1)
        samples_to_slice = Labels(
            names=["structure"],
            values=structures_to_keep,
        )
        sliced_tensorblock = fn.slice_block(
            tensorblock,
            samples_to_slice=samples_to_slice,
        )
        # Check 1: returned tensorblock has no values
        self.assertTrue(len(sliced_tensorblock.values.flatten()) == 0)

        # Check 2: returned tensorblock has dimension zero for samples
        self.assertTrue(sliced_tensorblock.values.shape[0] == 0)

        # Check 3: returned tensorblock has original dimension for properties
        self.assertEqual(
            sliced_tensorblock.values.shape[-1], tensorblock.values.shape[-1]
        )

    def test_slice_samples_tensormap_empty_no_gradients(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Remove the gradients to simplify test
        tensormap = fn.remove_gradients(tensormap)
        # Slice only 'structures' -1 (i.e. a sample that doesn't exist in the data)
        structures_to_keep = np.array([-1]).reshape(-1, 1)
        samples_to_slice = Labels(
            names=["structure"],
            values=structures_to_keep,
        )
        sliced_tensormap = fn.slice(
            tensormap,
            samples_to_slice=samples_to_slice,
        )
        for _, block in sliced_tensormap:
            # Check 1: all blocks are empty
            self.assertEqual(len(block.values.flatten()), 0)

        # Check 2: all the original keys are kept in the output tensormap
        self.assertTrue(
            np.all([key in sliced_tensormap.keys for key in tensormap.keys])
        )

    # TEST BLOCK 2: SLICING SAMPLES WITH GRADIENTS

    def test_slice_samples_tensorblock_gradients(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Define a single block to test
        tensorblock = tensormap.block(0)
        # Slice only 'structures' 2, 4, 6, 8
        structures_to_keep = np.arange(2, 10, 2).reshape(-1, 1)
        samples_to_slice = Labels(
            names=["structure"],
            values=structures_to_keep,
        )
        sliced_tensorblock = fn.slice_block(
            tensorblock,
            samples_to_slice=samples_to_slice,
        )
        for i, (parameter, gradient) in enumerate(sliced_tensorblock.gradients()):
            # Check 1: no slicing of properties has occurred
            self.assertEqual(
                list(tensorblock.gradients())[i][1].properties.asarray().shape,
                gradient.properties.asarray().shape,
            )
            # Check 2: samples have been sliced to the correct dimension
            self.assertEqual(
                len(sliced_tensorblock.samples),
                len(
                    [
                        struct_i
                        for struct_i in tensorblock.samples["structure"]
                        if struct_i in structures_to_keep
                    ]
                ),
            )
            # Check 3: samples in sliced block only feature desired structure indices
            self.assertTrue(
                np.all(
                    [
                        struct_i in structures_to_keep
                        for struct_i in sliced_tensorblock.samples["structure"]
                    ]
                )
            )
            # Check 4: same number of components as original
            self.assertEqual(
                len(gradient.components),
                len(list(tensorblock.gradients())[i][1].components),
            )

    def test_slice_samples_tensorblock_empty_gradients(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Define a single block to test
        tensorblock = tensormap.block(0)
        # Slice only 'structures' -1 (i.e. a sample that doesn't exist in the data)
        structures_to_keep = np.array([-1]).reshape(-1, 1)
        samples_to_slice = Labels(
            names=["structure"],
            values=structures_to_keep,
        )
        sliced_tensorblock = fn.slice_block(
            tensorblock,
            samples_to_slice=samples_to_slice,
        )
        # Check 1: returned tensorblock has no values
        self.assertEqual(len(sliced_tensorblock.values.flatten()), 0)

        for i, (_, gradient) in enumerate(sliced_tensorblock.gradients()):
            # Check 2: all gradients have no values
            self.assertEqual(len(gradient.data), 0)

            # Check 3: the shape of the Gradient values is equivalent to the
            # original in all but the samples (i.e. 1st) dimension
            self.assertEqual(
                gradient.data.shape[1:],
                list(tensorblock.gradients())[i][1].data.shape[1:],
            )

    def test_slice_samples_tensormap_empty_gradients(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Slice only 'structures' -1 (i.e. a sample that doesn't exist in the data)
        structures_to_keep = np.array([-1]).reshape(-1, 1)
        samples_to_slice = Labels(
            names=["structure"],
            values=structures_to_keep,
        )
        sliced_tensormap = fn.slice(
            tensormap,
            samples_to_slice=samples_to_slice,
        )
        for i, (_, block) in enumerate(sliced_tensormap):
            for j, (_, gradient) in enumerate(block.gradients()):
                # Check 1: all gradient blocks are empty
                self.assertEqual(len(gradient.data.flatten()), 0)

                # Check 2: the shape of the Gradient values is equivalent to the
                # original in all but the samples (i.e. 1st) dimension
                self.assertEqual(
                    gradient.data.shape[1:],
                    list(tensormap.block(i).gradients())[j][1].data.shape[1:],
                )

        # Check 3: all the original keys are kept in the output tensormap
        self.assertTrue(
            np.all([key in sliced_tensormap.keys for key in tensormap.keys])
        )

    # TEST BLOCK 3: SLICING PROPERTIES WITHOUT GRADIENTS

    def test_slice_properties_tensorblock_no_gradients(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Remove the gradients to simplify test
        tensormap = fn.remove_gradients(tensormap)
        # Define a single block to test
        tensorblock = tensormap.block(0)
        # Slice only 'n' (i.e. radial channels) 1, 3
        channels_to_keep = np.arange(1, 5, 2).reshape(-1, 1)
        properties_to_slice = Labels(
            names=["n"],
            values=channels_to_keep,
        )
        sliced_tensorblock = fn.slice_block(
            tensorblock,
            properties_to_slice=properties_to_slice,
        )
        # Check 1: no slicing of samples has occurred
        self.assertEqual(
            tensorblock.samples.asarray().shape,
            sliced_tensorblock.samples.asarray().shape,
        )
        # Check 2: properties have been sliced to the correct dimension
        self.assertEqual(
            len(sliced_tensorblock.properties),
            len(
                [
                    channel_i
                    for channel_i in tensorblock.properties["n"]
                    if channel_i in channels_to_keep
                ]
            ),
        )
        # Check 3: properties in sliced block only feature desired channel indices
        self.assertTrue(
            np.all(
                [
                    channel_i in channels_to_keep
                    for channel_i in sliced_tensorblock.properties["n"]
                ]
            )
        )
        # Check 4: no components have been sliced
        for i, comp in enumerate(sliced_tensorblock.components):
            self.assertEqual(len(comp), len(tensorblock.components[i]))

    def test_slice_properties_tensormap_no_gradients(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Remove the gradients to simplify test
        tensormap = fn.remove_gradients(tensormap)
        # Slice only 'n' (i.e. radial channels) 1, 3
        channels_to_keep = np.arange(1, 5, 2).reshape(-1, 1)
        properties_to_slice = Labels(
            names=["n"],
            values=channels_to_keep,
        )
        sliced_tensormap = fn.slice(
            tensormap,
            properties_to_slice=properties_to_slice,
        )
        for i, (key, block) in enumerate(tensormap):
            # Check 1: no slicing of samples has occurred
            self.assertEqual(
                sliced_tensormap.block(i).samples.asarray().shape,
                block.samples.asarray().shape,
            )
            # Check 2: properties have been sliced to the correct dimension
            self.assertEqual(
                len(sliced_tensormap.block(i).properties),
                len(
                    [
                        channel_i
                        for channel_i in block.properties["n"]
                        if channel_i in channels_to_keep
                    ]
                ),
            )
            # Check 3: properties in sliced block only feature desired channel indices
            self.assertTrue(
                np.all(
                    [
                        channel_i in channels_to_keep
                        for channel_i in sliced_tensormap.block(i).properties["n"]
                    ]
                )
            )
            # Check 4: no components have been sliced
            for j, comp in enumerate(block.components):
                self.assertEqual(
                    len(comp), len(sliced_tensormap.block(i).components[j])
                )

        # Check 5: all the keys in the sliced tensormap are in the original
        self.assertTrue(
            np.all([key in tensormap.keys for key in sliced_tensormap.keys])
        )

    def test_slice_properties_tensorblock_empty_no_gradients(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Remove the gradients to simplify test
        tensormap = fn.remove_gradients(tensormap)
        # Define a single block to test
        tensorblock = tensormap.block(0)
        # Slice only 'n' (i.e. radial channels) -1 (i.e. non-existent channel)
        channels_to_keep = np.array([-1]).reshape(-1, 1)
        properties_to_slice = Labels(
            names=["n"],
            values=channels_to_keep,
        )
        sliced_tensorblock = fn.slice_block(
            tensorblock,
            properties_to_slice=properties_to_slice,
        )
        # Check 1: returned tensorblock has no values
        self.assertEqual(len(sliced_tensorblock.values.flatten()), 0)

        # Check 2: returned tensorblock has dimension zero for properties
        self.assertEqual(sliced_tensorblock.values.shape[-1], 0)

        # Check 3: returned tensorblock has original dimension for samples
        self.assertEqual(
            sliced_tensorblock.values.shape[0], tensorblock.values.shape[0]
        )

    def test_slice_properties_tensormap_empty_no_gradients(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Remove the gradients to simplify test
        tensormap = fn.remove_gradients(tensormap)
        # Slice only 'n' (i.e. radial channels) -1 (i.e. non-existent channel)
        channels_to_keep = np.array([-1]).reshape(-1, 1)
        properties_to_slice = Labels(
            names=["n"],
            values=channels_to_keep,
        )
        sliced_tensormap = fn.slice(
            tensormap,
            properties_to_slice=properties_to_slice,
        )
        for _, block in sliced_tensormap:
            # Check 1: all blocks are empty
            self.assertEqual(len(block.values.flatten()), 0)

            # Check 2: the properties dimension is zero
            self.assertEqual(block.values.shape[-1], 0)

        # Check 2: all the original keys are kept in the output tensormap
        self.assertTrue(
            np.all([key in sliced_tensormap.keys for key in tensormap.keys])
        )

    # TEST BLOCK 4: SLICING PROPERTIES WITH GRADIENTS

    def test_slice_properties_tensorblock_gradients(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Define a single block to test
        tensorblock = tensormap.block(0)
        # Slice only 'n' (i.e. radial channels) 1, 3
        channels_to_keep = np.arange(1, 5, 2).reshape(-1, 1)
        properties_to_slice = Labels(
            names=["n"],
            values=channels_to_keep,
        )
        sliced_tensorblock = fn.slice_block(
            tensorblock,
            properties_to_slice=properties_to_slice,
        )
        for i, (parameter, gradient) in enumerate(sliced_tensorblock.gradients()):
            # Check 1: no slicing of samples has occurred
            self.assertEqual(
                list(tensorblock.gradients())[i][1].samples.asarray().shape,
                gradient.samples.asarray().shape,
            )
            # Check 2: properties have been sliced to the correct dimension
            self.assertEqual(
                len(sliced_tensorblock.properties),
                len(
                    [
                        channel_i
                        for channel_i in tensorblock.properties["n"]
                        if channel_i in channels_to_keep
                    ]
                ),
            )
            # Check 3: properties in sliced block only feature desired channel indices
            self.assertTrue(
                np.all(
                    [
                        channel_i in channels_to_keep
                        for channel_i in sliced_tensorblock.properties["n"]
                    ]
                )
            )
            # Check 4: same number of components as original
            self.assertEqual(
                len(gradient.components),
                len(list(tensorblock.gradients())[i][1].components),
            )

    def test_slice_properties_tensorblock_empty_gradients(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Define a single block to test
        tensorblock = tensormap.block(0)
        # Slice only 'n' (i.e. radial channels) -1 (i.e. non-existent channel)
        channels_to_keep = np.array([-1]).reshape(-1, 1)
        properties_to_slice = Labels(
            names=["n"],
            values=channels_to_keep,
        )
        sliced_tensorblock = fn.slice_block(
            tensorblock,
            properties_to_slice=properties_to_slice,
        )
        # Check 1: returned tensorblock has no values
        self.assertEqual(len(sliced_tensorblock.values.flatten()), 0)

        for i, (_, gradient) in enumerate(sliced_tensorblock.gradients()):
            # Check 2: all gradients have no values
            self.assertEqual(len(gradient.data.flatten()), 0)

            # Check 3: the shape of the Gradient values is equivalent to the
            # original in all but the properties (i.e. last) dimension
            self.assertEqual(
                gradient.data.shape[:-1],
                list(tensorblock.gradients())[i][1].data.shape[:-1],
            )

    def test_slice_properties_tensormap_empty_gradients(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Slice only 'n' (i.e. radial channels) -1 (i.e. non-existent channel)
        channels_to_keep = np.array([-1]).reshape(-1, 1)
        properties_to_slice = Labels(
            names=["n"],
            values=channels_to_keep,
        )
        sliced_tensormap = fn.slice(
            tensormap,
            properties_to_slice=properties_to_slice,
        )
        for i, (_, block) in enumerate(sliced_tensormap):
            for j, (_, gradient) in enumerate(block.gradients()):
                # Check 1: all gradient blocks are empty
                self.assertEqual(len(gradient.data.flatten()), 0)

                # Check 2: the shape of the Gradient values is equivalent to the
                # original in all but the properties (i.e. last) dimension
                self.assertEqual(
                    gradient.data.shape[:-1],
                    list(tensormap.block(i).gradients())[j][1].data.shape[:-1],
                )

        # Check 3: all the original keys are kept in the output tensormap
        self.assertTrue(
            np.all([key in sliced_tensormap.keys for key in tensormap.keys])
        )

    # TEST BLOCK 5: SLICING SAMPLES AND PROPERTIES SIMULTANEOUSLY

    def test_slice_samples_properties_tensorblock(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        tensorblock = tensormap.block(5)
        # Slice 'center' 1, 3, 5
        centers_to_keep = np.arange(1, 7, 2).reshape(-1, 1)
        samples_to_slice = Labels(
            names=["center"],
            values=centers_to_keep,
        )
        # Slice 'n' (i.e. radial channel) 0, 1, 2
        channels_to_keep = np.arange(0, 3).reshape(-1, 1)
        properties_to_slice = Labels(
            names=["n"],
            values=channels_to_keep,
        )
        sliced_tensorblock = fn.slice_block(
            tensorblock,
            samples_to_slice=samples_to_slice,
            properties_to_slice=properties_to_slice,
        )
        # Check 1: only desired centers indices are in the output.
        self.assertTrue(
            np.all(
                [
                    center_i in centers_to_keep
                    for center_i in sliced_tensorblock.samples["center"]
                ]
            )
        )
        # Check 2: only desired centers indices are in the output.
        self.assertTrue(
            np.all(
                [
                    channel_i in channels_to_keep
                    for channel_i in sliced_tensorblock.properties["n"]
                ]
            )
        )
        # Check 3: There are the correct number of samples
        self.assertEqual(
            sliced_tensorblock.values.shape[0],
            len(
                [
                    sample
                    for sample in tensorblock.samples
                    if sample["center"] in centers_to_keep
                ]
            ),
        )
        # Check 4: There are the correct number of properties
        self.assertEqual(
            sliced_tensorblock.values.shape[-1],
            len(
                [
                    prop
                    for prop in tensorblock.properties
                    if prop["n"] in channels_to_keep
                ]
            ),
        )
        # Check 6: actual values are what they should be
        samples_filter = [
            sample["center"] in centers_to_keep for sample in tensorblock.samples
        ]
        properties_filter = [
            prop["n"] in channels_to_keep for prop in tensorblock.properties
        ]
        self.assertTrue(
            np.array_equal(
                tensorblock.values[samples_filter][..., properties_filter],
                sliced_tensorblock.values,
            )
        )

    # TEST BLOCK 6: TESTING TYPE HANDLING OF USER-FACING SLICE FUNCTION

    def test_slice_type_handling(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Define a single block to test
        tensorblock = tensormap.block(0)
        # Slice 'center' 1, 3, 5
        structures_to_keep = np.arange(1, 7, 2).reshape(-1, 1)
        samples_to_slice = Labels(
            names=["center"],
            values=structures_to_keep,
        )
        # Check 1: TensorBlock -> TypeError
        with self.assertRaises(TypeError):
            fn.slice(tensorblock, samples_to_slice=samples_to_slice),
        # Check 2: TensorMap -> TensorMap
        self.assertTrue(
            isinstance(
                fn.slice(tensormap, samples_to_slice=samples_to_slice), TensorMap
            )
        )
        # Check 3: passing tensor=float raises TypeError
        with self.assertRaises(TypeError):
            fn.slice(5.0, samples_to_slice=samples_to_slice)
        # Check 4: passing samples_to_slice=np.array raises TypeError
        with self.assertRaises(TypeError):
            fn.slice(
                tensormap,
                samples_to_slice=np.array(
                    [
                        [
                            5,
                        ],
                        [
                            6,
                        ],
                    ]
                ),
            )

    def test_slice_block_type_handling(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Define a single block to test
        tensorblock = tensormap.block(0)
        # Slice 'center' 1, 3, 5
        structures_to_keep = np.arange(1, 7, 2).reshape(-1, 1)
        samples_to_slice = Labels(
            names=["center"],
            values=structures_to_keep,
        )
        # Check 1: TensorMap -> TypeError
        with self.assertRaises(TypeError):
            fn.slice_block(tensormap, samples_to_slice=samples_to_slice),
        # Check 2: TensorBlock -> TensorBlock
        self.assertTrue(
            isinstance(
                fn.slice_block(tensorblock, samples_to_slice=samples_to_slice),
                TensorBlock,
            )
        )
        # Check 3: passing tensor=float raises TypeError
        with self.assertRaises(TypeError):
            fn.slice_block(5.0, samples_to_slice=samples_to_slice)
        # Check 4: passing samples_to_slice=np.array raises TypeError
        with self.assertRaises(TypeError):
            fn.slice_block(
                tensorblock,
                samples_to_slice=np.array(
                    [
                        [
                            5,
                        ],
                        [
                            6,
                        ],
                    ]
                ),
            )

    # TEST BLOCK 7: TESTING WARNINGS OF THE SLICE FUNCTION

    def test_slice_samples_tensormap_partially_empty_warning(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Remove the gradients to simplify test
        tensormap = fn.remove_gradients(tensormap)
        # Slice only 'structures' 2
        structures_to_keep = np.array([2]).reshape(-1, 1)
        samples_to_slice = Labels(
            names=["structure"],
            values=structures_to_keep,
        )
        with self.assertWarns(UserWarning) as cm:
            # Check 1: warning raised as some empty blocks produced
            fn.slice(
                tensormap,
                samples_to_slice=samples_to_slice,
            )
        self.assertTrue(
            "Some TensorBlocks in the sliced TensorMap are now empty" in str(cm.warning)
        )

    def test_slice_samples_tensormap_completely_empty_warning(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Remove the gradients to simplify test
        tensormap = fn.remove_gradients(tensormap)
        # Slice only 'structures' -1 (i.e. a sample that doesn't exist in the data)
        structures_to_keep = np.array([-1]).reshape(-1, 1)
        samples_to_slice = Labels(
            names=["structure"],
            values=structures_to_keep,
        )
        with self.assertWarns(UserWarning) as cm:
            # Check 1: warning raised as all empty blocks produced
            fn.slice(
                tensormap,
                samples_to_slice=samples_to_slice,
            )
        self.assertTrue(
            "All TensorBlocks in the sliced TensorMap are now empty" in str(cm.warning)
        )

    def test_slice_properties_tensormap_completely_empty_warning(self):
        tensormap = equistore.io.load(
            os.path.join(DATA_ROOT, TEST_FILE),
            use_numpy=True,
        )
        # Remove the gradients to simplify test
        tensormap = fn.remove_gradients(tensormap)
        # Slice only 'n' (i.e. radial channels) -1 (i.e. non-existent channel)
        channels_to_keep = np.array([-1]).reshape(-1, 1)
        properties_to_slice = Labels(
            names=["n"],
            values=channels_to_keep,
        )
        with self.assertWarns(UserWarning) as cm:
            # Check 1: warning raised as all empty blocks produced
            fn.slice(
                tensormap,
                properties_to_slice=properties_to_slice,
            )
        self.assertTrue(
            "All TensorBlocks in the sliced TensorMap are now empty" in str(cm.warning)
        )


if __name__ == "__main__":
    unittest.main()
