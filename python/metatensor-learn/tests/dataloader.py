# """
# Module for testing the DataLoader class in :py:module:`dataloader`.
# """
import numpy as np
import pytest


torch = pytest.importorskip("torch")

import metatensor  # noqa: E402
from metatensor import TensorMap  # noqa: E402
from metatensor.learn.data import DataLoader, group  # noqa: E402

from . import utils  # noqa: E402


SAMPLE_INDICES = [i * 7 for i in range(6)]  # non-continuous sample index range


@pytest.mark.parametrize(
    "create_dataset",
    [utils.dataset_in_mem, utils.dataset_on_disk, utils.dataset_mixed_mem_disk],
)
def test_dataloader_indices(create_dataset, tmpdir):
    """
    Tests that the correct sample indices are loaded. As we are using the
    standard `Dataset` clas, the sample indices must be extracted from the
    TensorMaps.
    """
    with tmpdir.as_cwd():
        batch_size = 3
        dset = create_dataset(SAMPLE_INDICES)
        expected_indices = np.arange(len(SAMPLE_INDICES)).reshape(-1, batch_size)

        # Build dataloader from a random subset of sample indices
        dloader = DataLoader(dset, batch_size=batch_size, shuffle=False)

        for exp_idxs, batch in zip(expected_indices, dloader):
            assert (
                np.sort(
                    metatensor.unique_metadata(
                        batch.input, "samples", "sample_index"
                    ).values.reshape(-1)
                ).tolist()
                == np.sort(exp_idxs).tolist()
            )


@pytest.mark.parametrize(
    "create_dataset",
    [
        utils.indexed_dataset_in_mem,
        utils.indexed_dataset_on_disk,
        utils.indexed_dataset_mixed_mem_disk,
    ],
)
def test_dataloader_indices_indexed(create_dataset, tmpdir):
    """
    Tests that the correct sample indices are loaded. As we are using the
    standard `Dataset` clas, the sample indices must be extracted from the
    TensorMaps.
    """
    with tmpdir.as_cwd():
        batch_size = 3
        dset = create_dataset(SAMPLE_INDICES)
        expected_indices = np.array(SAMPLE_INDICES).reshape(-1, batch_size)

        # Build dataloader from a random subset of sample indices
        dloader = DataLoader(dset, batch_size=batch_size, shuffle=False)

        for exp_idxs, batch in zip(expected_indices, dloader):
            assert (
                np.sort(
                    metatensor.unique_metadata(
                        batch.input, "samples", "sample_index"
                    ).values.reshape(-1)
                ).tolist()
                == np.sort(exp_idxs).tolist()
            )


@pytest.mark.parametrize(
    "create_dataset",
    [
        utils.indexed_dataset_in_mem,
        utils.indexed_dataset_on_disk,
        utils.indexed_dataset_mixed_mem_disk,
    ],
)
def test_indexed_dataloader_indices_shuffle(create_dataset, tmpdir):
    """
    Tests the non-shuffling of the sample indices using the "group" collate
    function.
    """
    with tmpdir.as_cwd():
        dset = create_dataset(SAMPLE_INDICES)
        dloader = DataLoader(dset, collate_fn=group, batch_size=2, shuffle=False)

        batched_indices = np.array(SAMPLE_INDICES).reshape(-1, 2)

        matching = []
        for batch_i, batch in enumerate(dloader):
            matching.append(np.all(batch.sample_id == batched_indices[batch_i]))

        assert np.all(matching)


def test_dataloader_collate_fn_group_and_join(tmpdir):
    """
    Tests the output using the default "group_and_join" collate function.
    """
    with tmpdir.as_cwd():
        dset = utils.indexed_dataset_on_disk(SAMPLE_INDICES)

        # Build dataloader using default collate function "group_and_join"
        dloader = DataLoader(dset, batch_size=6)

        for batch in dloader:
            for field in [batch.input, batch.output, batch.auxiliary]:
                assert isinstance(field, TensorMap)
                assert all(
                    [
                        A
                        in metatensor.unique_metadata(
                            field, "samples", "sample_index"
                        ).values
                        for A in batch.sample_id
                    ]
                )


def test_dataloader_collate_fn_group(tmpdir):
    """
    Tests the output using the default "group_and_join" collate function.
    """
    with tmpdir.as_cwd():
        dset = utils.indexed_dataset_mixed_mem_disk(SAMPLE_INDICES)
        dloader = DataLoader(dset, collate_fn=group, batch_size=5)

        for batch in dloader:
            for field in [batch.input, batch.output, batch.auxiliary]:
                assert isinstance(field, tuple)
                for i, A in enumerate(batch.sample_id):
                    assert isinstance(field[i], TensorMap)
                    assert (
                        A
                        == metatensor.unique_metadata(
                            field[i], "samples", "sample_index"
                        ).values[0]
                    )
