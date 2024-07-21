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
            dataset = _tests_utils.indexed_dataset_in_mem(SAMPLE_INDICES)
        else:
            dataset = _tests_utils.dataset_in_mem(SAMPLE_INDICES)

        for idx, A in enumerate(SAMPLE_INDICES):
            # Check successful loading into memory upon calling __getitem__
            if indexed:
                assert isinstance(dataset.get_sample(A).input, TensorMap)
                assert isinstance(dataset.get_sample(A).output, TensorMap)
                assert isinstance(dataset.get_sample(A).auxiliary, TensorMap)
            else:
                assert isinstance(dataset[idx].input, TensorMap)
                assert isinstance(dataset[idx].output, TensorMap)
                assert isinstance(dataset[idx].auxiliary, TensorMap)


@pytest.mark.parametrize("indexed", [False, True])
def test_dataset_all_on_disk(indexed, tmpdir):
    """Test dataset that all data is on disk (i.e. a callable
    and not yet loaded into memory"""
    with tmpdir.as_cwd():
        if indexed:
            dataset = _tests_utils.indexed_dataset_on_disk(SAMPLE_INDICES)
        else:
            dataset = _tests_utils.dataset_on_disk(SAMPLE_INDICES)
        for idx, A in enumerate(SAMPLE_INDICES):
            # Check that callables are stored internally
            assert callable(dataset._data["input"])
            assert callable(dataset._data["output"])
            assert callable(dataset._data["auxiliary"])

            # Check successful loading into memory upon calling __getitem__
            if indexed:
                assert isinstance(dataset.get_sample(A).input, TensorMap)
                assert isinstance(dataset.get_sample(A).output, TensorMap)
                assert isinstance(dataset.get_sample(A).auxiliary, TensorMap)
            else:
                assert isinstance(dataset[idx].input, TensorMap)
                assert isinstance(dataset[idx].output, TensorMap)
                assert isinstance(dataset[idx].auxiliary, TensorMap)


@pytest.mark.parametrize("indexed", [False, True])
def test_dataset_mixed_mem_disk(indexed, tmpdir):
    """Test dataset that appropriate data is on disk (i.e. a callable)
    and not yet loaded into memory, or in memeory."""
    with tmpdir.as_cwd():
        if indexed:
            dataset = _tests_utils.indexed_dataset_mixed_mem_disk(SAMPLE_INDICES)
        else:
            dataset = _tests_utils.dataset_mixed_mem_disk(SAMPLE_INDICES)
        for idx, A in enumerate(SAMPLE_INDICES):
            # Check input/output data are stored as TensorMaps internally
            assert isinstance(dataset._data["input"][idx], TensorMap)
            assert isinstance(dataset._data["output"][idx], TensorMap)

            # Check auxiliary data is stored as a callable internally
            assert callable(dataset._data["auxiliary"])

            # Check successful loading into memory upon calling __getitem__
            if indexed:
                assert isinstance(dataset.get_sample(A).input, TensorMap)
                assert isinstance(dataset.get_sample(A).output, TensorMap)
                assert isinstance(dataset.get_sample(A).auxiliary, TensorMap)
            else:
                assert isinstance(dataset[idx].input, TensorMap)
                assert isinstance(dataset[idx].output, TensorMap)
                assert isinstance(dataset[idx].auxiliary, TensorMap)


def test_dataset_fields():
    """
    Tests that the dataset fields passed as kwargs are correctly set and
    returned in the correct order.
    """
    dataset = Dataset(
        a=[torch.zeros((1, 1)) for _ in range(10)],
        c=[torch.zeros((1, 1)) for _ in range(10)],
        b=[torch.zeros((1, 1)) for _ in range(10)],
    )
    assert np.all(dataset[5]._fields == ("a", "c", "b"))


def test_indexed_dataset_fields():
    """
    Tests that the indexed dataset fields passed as kwargs are correctly set and
    returned in the correct order.
    """
    dataset = IndexedDataset(
        a=[torch.zeros((1, 1)) for _ in range(10)],
        c=[torch.zeros((1, 1)) for _ in range(10)],
        sample_id=list(range(10)),
        b=[torch.zeros((1, 1)) for _ in range(10)],
    )
    assert np.all(dataset[5]._fields == ("sample_id", "a", "c", "b"))


def test_dataset_fields_arbitrary_types():
    """
    Tests that the dataset fields passed as arbitrary objects are correctly set
    and stored.
    """
    dataset = Dataset(
        a=["hello" for _ in range(10)],
        c=[(3, 4, 7) for _ in range(10)],
        b=[0 for _ in range(10)],
    )
    assert dataset[5].a == "hello"
    assert dataset[5].c == (3, 4, 7)
    assert dataset[5].b == 0
    assert np.all(dataset[5]._fields == ("a", "c", "b"))


def test_indexed_dataset_fields_arbitrary_types():
    """
    Tests that the dataset fields passed as arbitrary objects are correctly set
    and stored.
    """
    dataset = IndexedDataset(
        a=["hello" for _ in range(10)],
        c=[(3, 4, 7) for _ in range(10)],
        sample_id=list(range(10)),
        b=[0 for _ in range(10)],
    )
    assert dataset[5].a == "hello"
    assert dataset[5].c == (3, 4, 7)
    assert dataset[5].b == 0
    assert np.all(dataset[5]._fields == ("sample_id", "a", "c", "b"))


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
        dataset = create_dataset(SAMPLE_INDICES)
        message = "Index 1000 not in dataset"
        with pytest.raises(ValueError, match=message):
            dataset[1000]


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
        dataset = create_dataset(SAMPLE_INDICES)
        message = "Index 1000 not in dataset"
        with pytest.raises(ValueError, match=message):
            dataset[1000]

        message = "Sample ID 'foo' not in dataset"
        with pytest.raises(ValueError, match=message):
            dataset.get_sample("foo")


def test_dataset_callable_invalid_callable():
    """
    Tests that passing a data fields as a callable that is invalid with a
    specific sample idx raises the appropriate error.
    """
    dataset = Dataset(
        c=lambda y: metatensor.load(f"path/to/{y}"),
        size=1,
    )

    message = "Error loading data field 'c' at sample index 0"
    with pytest.raises(ValueError, match=message):
        dataset[0]


def test_indexed_dataset_invalid_callable():
    """
    Tests that passing a data fields as a callable that is invalid with a
    specific sample ID raises the appropriate error.
    """
    dataset = IndexedDataset(
        c=lambda y: metatensor.load(f"path/to/{y}"),
        sample_id=["cat", "dog"],
    )

    message = "Error loading data field 'c' for sample 'cat'"
    with pytest.raises(ValueError, match=message):
        dataset.get_sample("cat")


def test_dataset_inconsistent_lengths():
    """
    Tests that passing data fields as callables with no sample indices raises
    the appropriate error.
    """
    message = (
        "Number of samples inconsistent between argument 'size' (5) "
        "and data fields: ([10])"
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
        "and data fields: ([9, 10])"
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
    message = "Sample IDs must be unique, found some duplicate"
    with pytest.raises(ValueError, match=message):
        IndexedDataset(
            a=[1, 2, 3],
            sample_id=["a", "b", "a"],
        )


def test_iterate_over_dataset():
    """
    Tests iterating over the Dataset object.
    """
    dataset = Dataset(
        a=[torch.ones((1, 1)) * i for i in range(5)],
        c=[torch.zeros((1, 1)) * 10 for _ in range(5)],
        b=[torch.zeros((1, 1)) for _ in range(5)],
    )

    for i, sample in enumerate(dataset):
        assert sample.a == torch.ones((1, 1)) * i
        assert sample.c == torch.zeros((1, 1))
        assert sample.b == torch.zeros((1, 1))


def test_iterate_over_indexed_dataset():
    """
    Tests iterating over the IndexedDataset object.
    """
    sample_ids = ["dog", "cat", "fish", "bird", "mouse"]
    dataset = IndexedDataset(
        a=[torch.ones((1, 1)) * i for i in range(5)],
        c=[torch.zeros((1, 1)) * 10 for _ in range(5)],
        b=[torch.zeros((1, 1)) for _ in range(5)],
        sample_id=sample_ids,
    )

    for i, sample in enumerate(dataset):
        assert sample.sample_id == sample_ids[i]
        assert sample.a == torch.ones((1, 1)) * i
        assert sample.c == torch.zeros((1, 1))
        assert sample.b == torch.zeros((1, 1))


def test_dataset_non_ident_names():
    dataset = Dataset.from_dict(
        {
            "foo bar": [1, 2, 3],
            "__underscore": [0, 2, 4],
            "class": [-1, -2, -3],
        },
    )

    for i, sample in enumerate(dataset):
        assert sample["foo bar"] == i + 1
        assert sample.__underscore == 2 * i
        assert sample["__underscore"] == 2 * i
        assert sample["class"] == -(i + 1)

    assert str(sample) == "Sample('foo bar'=3, __underscore=4, 'class'=-3)"
    assert sample._asdict() == {"foo bar": 3, "__underscore": 4, "class": -3}

    dataset = IndexedDataset.from_dict(
        {
            "foo bar": [1, 2, 3],
            "__underscore": [0, 2, 4],
            "class": [-1, -2, -3],
        },
        sample_id=[0, 1, 2],
    )

    for i, sample in enumerate(dataset):
        assert sample.sample_id == i
        assert sample["foo bar"] == i + 1
        assert sample.__underscore == 2 * i
        assert sample["__underscore"] == 2 * i
        assert sample["class"] == -(i + 1)

    assert str(sample) == "Sample(sample_id=2, 'foo bar'=3, __underscore=4, 'class'=-3)"
    assert sample._asdict() == {
        "sample_id": 2,
        "foo bar": 3,
        "__underscore": 4,
        "class": -3,
    }
