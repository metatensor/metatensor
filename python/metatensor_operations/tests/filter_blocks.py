import os

import numpy as np
import pytest

import metatensor
from metatensor import Labels, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def tensor() -> TensorMap:
    return metatensor.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.mts"))


def test_keep_all(tensor):
    full_tensor = metatensor.filter_blocks(tensor, tensor.keys)
    assert metatensor.equal(tensor, full_tensor)


def test_keep_none(tensor):
    tensor = metatensor.filter_blocks(tensor, Labels.empty(tensor.keys.names))
    empty_tensor = TensorMap(keys=Labels.empty(tensor.keys.names), blocks=[])
    assert metatensor.equal(tensor, empty_tensor)


def test_keep_selection(tensor):
    assert np.unique(tensor.keys["center_type"]).tolist() == [1, 6, 8]

    keys_to_keep = Labels("center_type", np.array([[1]]))
    new_tensor = metatensor.filter_blocks(tensor, keys_to_keep)

    assert np.unique(new_tensor.keys["center_type"]).tolist() == [1]


def test_keep_two_random_keys(tensor):
    # test the behavior when two random keys are dropped
    n_blocks = len(tensor.keys)
    indices_to_keep = np.random.choice(list(range(n_blocks)), size=2, replace=False)

    values_to_keep = [tensor.keys[i].values for i in indices_to_keep]
    keys_to_keep = Labels(tensor.keys.names, np.vstack(values_to_keep))

    values_to_keep = [
        tensor.keys[i].values for i in range(n_blocks) if i not in indices_to_keep
    ]

    ref_blocks = [tensor[key].copy() for key in keys_to_keep]
    ref_tensor = TensorMap(keys_to_keep, ref_blocks)

    new_tensor = metatensor.filter_blocks(tensor, keys_to_keep)
    assert metatensor.equal(new_tensor, ref_tensor)


def test_larger_filter(tensor):
    """A filter containing more values than the actual tensor"""

    key_to_filter = Labels(
        tensor.keys.names,
        np.concatenate(
            [np.array([[1000] * len(tensor.keys.names)]), tensor.keys.values], axis=0
        ),
    )
    test_filter = metatensor.filter_blocks(tensor, key_to_filter)

    assert metatensor.equal(test_filter, tensor)


def test_copy_flag(tensor):
    """
    Checks that settings the copy flag to true or false gives the same result.
    Also checks that inplace modifications of the underlying data doesn't affect
    the copied returned tensor, but it does for the un-copied version.
    """
    # Define some keys to drop
    values_to_keep = [tensor.keys[i].values for i in range(5)]
    keys_to_keep = Labels(
        tensor.keys.names,
        np.vstack(values_to_keep),
    )

    # Drop blocks with copy=True flag
    new_tensor_copied = metatensor.filter_blocks(tensor, keys_to_keep, copy=True)

    # Drop blocks with copy=False flag
    new_tensor_not_copied = metatensor.filter_blocks(tensor, keys_to_keep, copy=False)

    # Check that the resulting tensor are equal whether or not they are copied
    assert metatensor.equal(new_tensor_copied, new_tensor_not_copied)

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
