"""
Module for testing the Dataset class in :py:module:`dataset`.
"""

import re

import numpy as np
import pytest


torch = pytest.importorskip("torch")


import metatensor  # noqa: E402
from metatensor import TensorMap  # noqa: E402
from metatensor.learn.data import Dataset, IndexedDataset  # noqa: E402

from . import _tests_utils  # noqa: E402


SAMPLE_INDICES = [i * 7 for i in range(6)]  # non-continuous sample index range


@pytest.mark.parametrize("indexed", [False, True])
def test_dataset_all_in_memory(indexed, tmpdir):
    """Tests dataset that all data is in memory"""
    with tmpdir.as_cwd():
        if indexed:
            dset = _tests_utils.indexed_dataset_in_mem(SAMPLE_INDICES)
        else:
            dset = _tests_utils.dataset_in_mem(SAMPLE_INDICES)

        for idx, A in enumerate(SAMPLE_INDICES):
            # Check successful loading into memory upon calling __getitem__
            if indexed:
                assert isinstance(dset.get_sample(A).input, TensorMap)
                assert isinstance(dset.get_sample(A).output, TensorMap)
                assert isinstance(dset.get_sample(A).auxiliary, TensorMap)
            else:
                assert isinstance(dset[idx].input, TensorMap)
                assert isinstance(dset[idx].output, TensorMap)
                assert isinstance(dset[idx].auxiliary, TensorMap)


@pytest.mark.parametrize("indexed", [False, True])
def test_dataset_all_on_disk(indexed, tmpdir):
    """Test dataset that all data is on disk (i.e. a callable
    and not yet loaded into memory"""
    with tmpdir.as_cwd():
        if indexed:
            dset = _tests_utils.indexed_dataset_on_disk(SAMPLE_INDICES)
        else:
            dset = _tests_utils.dataset_on_disk(SAMPLE_INDICES)
        for idx, A in enumerate(SAMPLE_INDICES):
            # Check that callables are stored internally
            assert callable(dset._data["input"])
            assert callable(dset._data["output"])
            assert callable(dset._data["auxiliary"])

            # Check successful loading into memory upon calling __getitem__
            if indexed:
                assert isinstance(dset.get_sample(A).input, TensorMap)
                assert isinstance(dset.get_sample(A).output, TensorMap)
                assert isinstance(dset.get_sample(A).auxiliary, TensorMap)
            else:
                assert isinstance(dset[idx].input, TensorMap)
                assert isinstance(dset[idx].output, TensorMap)
                assert isinstance(dset[idx].auxiliary, TensorMap)


@pytest.mark.parametrize("indexed", [False, True])
def test_dataset_mixed_mem_disk(indexed, tmpdir):
    """Test dataset that appropriate data is on disk (i.e. a callable)
    and not yet loaded into memory, or in memeory."""
    with tmpdir.as_cwd():
        if indexed:
            dset = _tests_utils.indexed_dataset_mixed_mem_disk(SAMPLE_INDICES)
        else:
            dset = _tests_utils.dataset_mixed_mem_disk(SAMPLE_INDICES)
        for idx, A in enumerate(SAMPLE_INDICES):
            # Check input/output data are stored as TensorMaps internally
            assert isinstance(dset._data["input"][idx], TensorMap)
            assert isinstance(dset._data["output"][idx], TensorMap)

            # Check auxiliary data is stored as a callable internally
            assert callable(dset._data["auxiliary"])

            # Check successful loading into memory upon calling __getitem__
            if indexed:
                assert isinstance(dset.get_sample(A).input, TensorMap)
                assert isinstance(dset.get_sample(A).output, TensorMap)
                assert isinstance(dset.get_sample(A).auxiliary, TensorMap)
            else:
                assert isinstance(dset[idx].input, TensorMap)
                assert isinstance(dset[idx].output, TensorMap)
                assert isinstance(dset[idx].auxiliary, TensorMap)


def test_dataset_fields():
    """
    Tests that the dataset fields passed as kwargs are correctly set and
    returned in the correct order.
    """
    dset = Dataset(
        a=[torch.zeros((1, 1)) for _ in range(10)],
        c=[torch.zeros((1, 1)) for _ in range(10)],
        b=[torch.zeros((1, 1)) for _ in range(10)],
    )
    assert np.all(dset[5]._fields == ("a", "c", "b"))


def test_indexed_dataset_fields():
    """
    Tests that the indexed dataset fields passed as kwargs are correctly set and
    returned in the correct order.
    """
    dset = IndexedDataset(
        a=[torch.zeros((1, 1)) for _ in range(10)],
        c=[torch.zeros((1, 1)) for _ in range(10)],
        sample_id=list(range(10)),
        b=[torch.zeros((1, 1)) for _ in range(10)],
    )
    assert np.all(dset[5]._fields == ("sample_id", "a", "c", "b"))


def test_dataset_fields_arbitrary_types():
    """
    Tests that the dataset fields passed as arbitrary objects are correctly set
    and stored.
    """
    dset = Dataset(
        a=["hello" for _ in range(10)],
        c=[(3, 4, 7) for _ in range(10)],
        b=[0 for _ in range(10)],
    )
    assert dset[5].a == "hello"
    assert dset[5].c == (3, 4, 7)
    assert dset[5].b == 0
    assert np.all(dset[5]._fields == ("a", "c", "b"))


def test_indexed_dataset_fields_arbitrary_types():
    """
    Tests that the dataset fields passed as arbitrary objects are correctly set
    and stored.
    """
    dset = IndexedDataset(
        a=["hello" for _ in range(10)],
        c=[(3, 4, 7) for _ in range(10)],
        sample_id=list(range(10)),
        b=[0 for _ in range(10)],
    )
    assert dset[5].a == "hello"
    assert dset[5].c == (3, 4, 7)
    assert dset[5].b == 0
    assert np.all(dset[5]._fields == ("sample_id", "a", "c", "b"))


@pytest.mark.parametrize(
    "create_dataset",
    [
        _tests_utils.dataset_in_mem,
        _tests_utils.dataset_on_disk,
        _tests_utils.dataset_mixed_mem_disk,
    ],
)
def test_dataset_nonexistant_index_error(create_dataset, tmpdir):
    """
    Tests that calling __getitem__ for a non-existant index raises an error.
    """
    with tmpdir.as_cwd():
        dset = create_dataset(SAMPLE_INDICES)
        message = "Index 1000 not in dataset"
        with pytest.raises(ValueError, match=message):
            dset[1000]


@pytest.mark.parametrize(
    "create_dataset",
    [
        _tests_utils.indexed_dataset_in_mem,
        _tests_utils.indexed_dataset_on_disk,
        _tests_utils.indexed_dataset_mixed_mem_disk,
    ],
)
def test_indexed_dataset_nonexistant_index_error(create_dataset, tmpdir):
    """
    Tests that calling __getitem__ for a non-existant index raises an error.
    """
    with tmpdir.as_cwd():
        dset = create_dataset(SAMPLE_INDICES)
        message = "Index 1000 not in dataset"
        with pytest.raises(ValueError, match=message):
            dset[1000]

        message = "Sample ID 'foo' not in dataset"
        with pytest.raises(ValueError, match=message):
            dset.get_sample("foo")


def test_dataset_callable_invalid_callable():
    """
    Tests that passing a data fields as a callable that is invalid with a
    specific numeric sample idx raises the appropriate error.
    """
    message = "Error loading data field 'c' at numeric sample index 0"
    with pytest.raises(IOError) as excinfo:
        dset = Dataset(
            c=lambda y: metatensor.load(f"path/to/{y}"),
            size=1,
        )
        dset[0]
    assert message in str(excinfo.value)


def test_indexed_dataset_invalid_callable():
    """
    Tests that passing a data fields as a callable that is invalid with a
    specific sample ID raises the appropriate error.
    """
    message = "Error loading data field 'c' for sample ID cat"
    with pytest.raises(IOError) as excinfo:
        dset = IndexedDataset(
            c=lambda y: metatensor.load(f"path/to/{y}"),
            sample_id=["cat", "dog"],
        )
        dset.get_sample("cat")
    assert message in str(excinfo.value)


def test_dataset_inconsistent_lengths():
    """
    Tests that passing data fields as callables with no sample indices raises
    the appropriate error.
    """
    message = (
        "Number of samples inconsistent between argument 'size' (5) "
        "and data fields in kwargs: ([10])"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        Dataset(
            a=lambda x: f"path/to/{x}",
            c=lambda y: f"path/to/{y}",
            x=list(range(10)),
            size=5,
        )

    message = "Number of samples inconsistent between data fields: [10, 9]"
    with pytest.raises(ValueError, match=re.escape(message)):
        Dataset(x=list(range(10)), y=list(range(9)))


def test_indexed_dataset_inconsistent_lengths():
    """
    Tests that passing data fields as callables with no sample indices raises
    the appropriate error.
    """
    message = (
        "Number of samples inconsistent between argument 'sample_id' (9) "
        "and data fields in kwargs: ([10])"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        IndexedDataset(
            a=lambda x: f"path/to/{x}",
            c=lambda y: f"path/to/{y}",
            x=list(range(10)),
            sample_id=list(range(9)),
        )


def test_indexed_dataset_nonunique_sample_id():
    """
    Tests the appropriate error is raised when non-unique sample IDs are
    passed.
    """
    message = "Sample IDs must be unique. Found duplicate sample IDs."
    with pytest.raises(ValueError, match=message):
        IndexedDataset(
            a=[1, 2, 3],
            sample_id=["a", "b", "a"],
        )


def test_iterate_over_dataset():
    """
    Tests iterating over the Dataset object.
    """
    dset = Dataset(
        a=[torch.ones((1, 1)) * i for i in range(5)],
        c=[torch.zeros((1, 1)) * 10 for _ in range(5)],
        b=[torch.zeros((1, 1)) for _ in range(5)],
    )

    for i, sample in enumerate(dset):
        assert sample.a == torch.ones((1, 1)) * i
        assert sample.c == torch.zeros((1, 1))
        assert sample.b == torch.zeros((1, 1))


def test_iterate_over_indexeddataset():
    """
    Tests iterating over the IndexedDataset object.
    """
    sample_ids = ["dog", "cat", "fish", "bird", "mouse"]
    dset = IndexedDataset(
        a=[torch.ones((1, 1)) * i for i in range(5)],
        c=[torch.zeros((1, 1)) * 10 for _ in range(5)],
        b=[torch.zeros((1, 1)) for _ in range(5)],
        sample_id=sample_ids,
    )

    for i, sample in enumerate(dset):
        assert sample.sample_id == sample_ids[i]
        assert sample.a == torch.ones((1, 1)) * i
        assert sample.c == torch.zeros((1, 1))
        assert sample.b == torch.zeros((1, 1))
