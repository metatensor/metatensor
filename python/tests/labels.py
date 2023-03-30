import numpy as np
import pytest
from numpy.testing import assert_equal
from utils import tensor_map

from equistore import EquistoreError, Labels


class TestLabels:
    def test_python_labels(self):
        labels = Labels(
            names=["a", "b"],
            values=np.array([[0, 0]]),
        )

        assert labels.names == ("a", "b")
        assert len(labels) == 1
        assert tuple(labels[0]) == (0, 0)

        assert_equal(labels.asarray(), np.array([[0, 0]]))

        # check we can use single str for single name
        labels = Labels(
            names=["a"],
            values=np.array([[1]]),
        )
        labels_str = Labels(
            names="a",
            values=np.array([[1]]),
        )

        assert labels.names == ("a",)
        assert labels.names == labels_str.names
        assert len(labels) == 1
        assert len(labels_str) == 1
        assert tuple(labels_str[0]) == (1,)
        assert tuple(labels_str[0]) == tuple(labels[0])

        assert_equal(labels.asarray(), np.array([[1]]))
        assert_equal(labels.asarray(), labels_str.asarray())

        # check empty arrays
        labels = Labels(
            names=[],
            values=np.array([[]], dtype=np.int32),
        )

        labels_str = Labels(
            names="",
            values=np.array([[]], dtype=np.int32),
        )

        assert labels.names == []
        assert labels.names == labels_str.names
        assert len(labels) == 1
        assert len(labels_str) == 1
        assert tuple(labels_str[0]) == tuple(labels[0])

        assert_equal(labels.asarray(), np.array([[]]))
        assert_equal(labels.asarray(), labels_str.asarray())

        # check that we can convert from more than strict 2D arrays of int32
        labels = Labels(
            names=["a", "b"],
            values=np.array([[0, 0]]),
        )

        labels = Labels(
            names=["a", "b"],
            values=np.array([[0, 0]], dtype=np.int64),
        )

        msg = "Labels values must be convertible to integers"
        with pytest.raises(TypeError, match=msg):
            labels = Labels(
                names=["a", "b"], values=np.array([[0, 0]], dtype=np.float64)
            )

    def test_native_labels(self):
        tensor = tensor_map()
        labels = tensor.keys

        assert labels.names == ("key_1", "key_2")
        assert len(labels) == 4
        assert tuple(labels[0]) == (0, 0)
        assert tuple(labels[1]) == (1, 0)
        assert tuple(labels[2]) == (2, 2)
        assert tuple(labels[3]) == (2, 3)

        expected = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 2],
                [2, 3],
            ]
        )
        assert_equal(labels.asarray(), expected)

    def test_position(self):
        tensor = tensor_map()
        labels = tensor.keys

        assert labels.position((0, 0)) == 0
        assert labels.position((2, 3)) == 3
        assert labels.position((2, -1)) is None

    def test_contains(self):
        tensor = tensor_map()
        labels = tensor.keys

        assert (0, 0) in labels
        assert (2, 3) in labels
        assert (2, -1) not in labels

    def test_named_tuples(self):
        tensor = tensor_map()
        labels = tensor.keys

        iterator = labels.as_namedtuples()
        first = next(iterator)

        assert isinstance(first, tuple)
        assert hasattr(first, "_fields")

        assert first.key_1 == 0
        assert first.key_2 == 0
        assert first.as_dict() == {"key_1": 0, "key_2": 0}

        assert first == (0, 0)
        assert next(iterator) == (1, 0)
        assert next(iterator) == (2, 2)
        assert next(iterator) == (2, 3)

        with pytest.raises(StopIteration):
            next(iterator)

    def test_not_writeable(self):
        tensor = tensor_map()
        labels = tensor.keys

        with pytest.raises(ValueError, match="assignment destination is read-only"):
            labels[0][0] = 4

    def test_invalid_names(self):
        msg = "invalid parameter: 'not an ident' is not a valid label name"
        with pytest.raises(EquistoreError, match=msg):
            Labels(
                names=["not an ident"],
                values=np.array([[0]]),
            )

    def test_zero_length_label(self):
        label = Labels(["sample", "structure", "atom"], np.array([]))
        assert len(label) == 0

    def test_labels_single(self):
        label = Labels.single()
        assert label.names == ("_",)
        assert label.shape == (1,)

    def test_labels_empty(self):
        names = (
            "foo",
            "bar",
            "baz",
        )
        label = Labels.empty(names)
        assert label.names == names
        assert len(label) == 0

    def test_labels_contiguous(self):
        labels = Labels(
            names=["a", "b"],
            values=np.arange(32).reshape(-1, 2),
        )

        labels = labels[::-1]
        ptr = labels._as_eqs_labels_t().values

        shape = (len(labels), len(labels.names))
        array = np.ctypeslib.as_array(ptr, shape=shape)
        array = array.view(dtype=labels.dtype).reshape(-1)

        assert np.all(array == labels)

    def test_arange_one_argument(self):
        labels_arange = Labels.arange("name", 10)
        assert labels_arange.asarray().shape == (10, 1)
        assert labels_arange.names == ("name",)
        np.testing.assert_equal(labels_arange.asarray().reshape((-1,)), np.arange(10))

    def test_arange_two_arguments(self):
        labels_arange = Labels.arange("dummy", 10, 42)
        assert labels_arange.names == ("dummy",)
        np.testing.assert_equal(labels_arange.asarray().reshape((-1,)), np.arange(10, 42))

    def test_arange_three_arguments(self):
        labels_arange = Labels.arange("samples", 0, 10, 2)
        assert labels_arange.names == ("samples",)
        np.testing.assert_equal(
            labels_arange.asarray().reshape((-1,)), np.arange(0, 10, 2)
        )

    def test_arange_incorrect_arguments(self):
        with pytest.raises(ValueError, match="3"):
            Labels.arange("dummy", 0, 10, 2, 4)
        with pytest.raises(ValueError, match="at least"):
            Labels.arange("dummy")
        with pytest.raises(EquistoreError, match="label name"):
            Labels.arange(0, 1, 2)
        with pytest.raises(ValueError, match="integer"):
            Labels.arange(0.0, 1.0, 2)
        with pytest.raises(ValueError, match="integer"):
            Labels.arange("dummy", 0, 5, 0.2)
