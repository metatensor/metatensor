import os

import numpy as np
import pytest

import metatensor as mts
from metatensor import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def tensor() -> TensorMap:
    return mts.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.mts"))


def test_drop_blocks_empty_tensor():
    """
    Tests that dropping blocks from an empty tensor returns an empty tensor.
    """
    empty_tensor = TensorMap(keys=Labels.empty(["a", "b"]), blocks=[])
    new_tensor = mts.drop_blocks(empty_tensor, Labels.empty(["a", "b"]))
    assert mts.equal(new_tensor, empty_tensor)


def test_drop_block_with_an_empty_dimension():
    """
    Define a TensorMap in which block_2 has a zero-length dimension.
    Assert then that calling drop_empty_blocks() removes block_2 from
    the resulting TensorMap.
    """
    block_1 = TensorBlock(
        values=np.full((5, 3), 2.0),
        samples=Labels.range("sample", 5),
        components=[],
        properties=Labels.range("property", 3),
    )
    block_2 = TensorBlock(
        values=np.full((0, 4), 1.0),
        samples=Labels.range("sample", 0),
        components=[],
        properties=Labels.range("property", 4),
    )
    block_3 = TensorBlock(
        values=np.full((2, 6), 3.0),
        samples=Labels.range("sample", 2),
        components=[],
        properties=Labels.range("property", 6),
    )
    keys = Labels(names=["id"], values=np.array([[0], [1], [2]]))

    bkp_block_1 = block_1.copy()
    bkp_block_3 = block_3.copy()

    # Create the TensorMap
    tensor = TensorMap(keys=keys, blocks=[block_1, block_2, block_3])

    # Drop blocks with empty dimensions
    new_tensor = mts.drop_empty_blocks(tensor, copy=False)
    assert len(new_tensor) == 2

    # Check which idxes were kept
    kept_idxs = new_tensor.keys.values.flatten().tolist()

    assert kept_idxs == [0, 2], (
        "Blocks with empty dimensions were not dropped correctly"
    )

    np.testing.assert_allclose(new_tensor[0].values, bkp_block_1.values)
    np.testing.assert_allclose(new_tensor[1].values, bkp_block_3.values)


def test_drop_empty_blocks_in_empty_tensor():
    """
    Tests that dropping empty blocks from an empty tensor returns an empty tensor.
    """
    empty_tensor = TensorMap(keys=Labels.empty(["key_1", "key_2"]), blocks=[])
    new_tensor = mts.drop_empty_blocks(empty_tensor, copy=False)
    assert mts.equal(new_tensor, empty_tensor)


def test_drop_empty_blocks_on_tensor_with_no_empty_blocks():
    """
    Define a TensorMap in which no block has a zero-length dimension.
    Assert than that calling drop_empty_blocks() leaves all blocks intact.
    """
    block_1 = TensorBlock(
        values=np.full((5, 3), 2.0),
        samples=Labels.range("sample", 5),
        components=[],
        properties=Labels.range("property", 3),
    )
    block_2 = TensorBlock(
        values=np.full((3, 4), 1.0),
        samples=Labels.range("sample", 3),
        components=[],
        properties=Labels.range("property", 4),
    )
    block_3 = TensorBlock(
        values=np.full((2, 6), 3.0),
        samples=Labels.range("sample", 2),
        components=[],
        properties=Labels.range("property", 6),
    )
    keys = Labels(names=["id"], values=np.array([[0], [1], [2]]))

    # Create the TensorMap
    tensor = TensorMap(keys=keys, blocks=[block_1, block_2, block_3])

    # Drop blocks with empty dimensions
    new_tensor = mts.drop_empty_blocks(tensor, copy=False)
    assert mts.equal(tensor, new_tensor)


def test_drop_all(tensor):
    tensor = mts.drop_blocks(tensor, tensor.keys)
    empty_tensor = TensorMap(keys=Labels.empty(tensor.keys.names), blocks=[])
    assert mts.equal(tensor, empty_tensor)


def test_drop_two_random_keys(tensor):
    # test the behavior when two random keys are dropped
    n_blocks = len(tensor.keys)
    indices_to_drop = np.random.choice(list(range(n_blocks)), size=2, replace=False)

    values_to_drop = [tensor.keys[i].values for i in indices_to_drop]
    keys_to_drop = Labels(tensor.keys.names, np.vstack(values_to_drop))

    values_to_keep = [
        tensor.keys[i].values for i in range(n_blocks) if i not in indices_to_drop
    ]
    keys_to_keep = Labels(tensor.keys.names, np.vstack(values_to_keep))

    ref_blocks = [tensor[key].copy() for key in keys_to_keep]
    ref_tensor = TensorMap(keys_to_keep, ref_blocks)

    new_tensor = mts.drop_blocks(tensor, keys_to_drop)
    assert mts.equal(new_tensor, ref_tensor)


def test_drop_selection(tensor):
    assert np.unique(tensor.keys["center_type"]).tolist() == [1, 6, 8]

    keys_to_drop = Labels("center_type", np.array([[1]]))
    new_tensor = mts.drop_blocks(tensor, keys_to_drop)

    assert np.unique(new_tensor.keys["center_type"]).tolist() == [6, 8]


def test_drop_nothing(tensor):
    empty_key = Labels.empty(tensor.keys.names)
    new_tensor = mts.drop_blocks(tensor, empty_key)
    assert new_tensor == tensor


def test_copy_flag(tensor):
    """
    Checks that settings the copy flag to true or false gives the same result.
    Also checks that inplace modifications of the underlying data doesn't affect
    the copied returned tensor, but it does for the un-copied version.
    """
    # Define some keys to drop
    values_to_drop = [tensor.keys[i].values for i in range(5)]
    keys_to_drop = Labels(
        tensor.keys.names,
        np.vstack(values_to_drop),
    )

    # Drop blocks with copy=True flag
    new_tensor_copied = mts.drop_blocks(tensor, keys_to_drop, copy=True)

    # Drop blocks with copy=False flag
    new_tensor_not_copied = mts.drop_blocks(tensor, keys_to_drop, copy=False)

    # Check that the resulting tensor are equal whether or not they are copied
    assert mts.equal(new_tensor_copied, new_tensor_not_copied)

    # Now modify the original tensor's block values in place
    for block in tensor.blocks():
        block.values[:] += 3.14

    # The copied tensor's values should not have been edited
    for key in new_tensor_copied.keys:
        assert np.all(new_tensor_copied[key].values != tensor[key].values)

    # But the un-copied tensor's values should have been
    for key in new_tensor_not_copied.keys:
        assert np.all(new_tensor_not_copied[key].values == tensor[key].values)
        assert np.all(
            new_tensor_not_copied[key].values == new_tensor_copied[key].values[:] + 3.14
        )
