import numpy as np
import pytest

from equistore.core import EquistoreError, Labels


def test_constructor():
    labels = Labels(names=("a", "b"), values=np.array([[0, 0]]))

    assert labels.names == ["a", "b"]
    assert len(labels) == 1
    assert tuple(labels[0]) == (0, 0)

    np.testing.assert_equal(labels.values, np.array([[0, 0]]))

    # check we can use single str for a single name
    labels = Labels(
        names="abc",
        values=np.array([[1]]),
    )

    assert labels.names == ["abc"]
    assert len(labels) == 1
    assert tuple(labels[0]) == (1,)

    np.testing.assert_equal(labels.values, np.array([[1]]))


def test_empty_labels():
    # labels without dimensions
    labels = Labels(names=[], values=np.array([[]], dtype=np.int32))
    labels_str = Labels(names="", values=np.array([[]], dtype=np.int32))

    assert labels.names == []
    assert labels.names == labels_str.names
    assert len(labels) == 0
    assert len(labels_str) == 0

    assert labels.values.shape == (0, 0)
    assert labels_str.values.shape == (0, 0)

    # labels without entries
    labels = Labels(names=["a", "b"], values=np.empty((0, 2)))
    assert labels.names == ["a", "b"]
    assert len(labels) == 0

    labels = Labels.empty(["a", "b"])
    assert labels.names == ["a", "b"]
    assert len(labels) == 0


def test_convert_types():
    """check that we can convert from more than strict 2D arrays of int32"""
    Labels(names=["a", "b"], values=np.array([[0, 0]], dtype=np.int64))


def test_wrong_types():
    msg = "Labels values must be convertible to integers"
    with pytest.raises(TypeError, match=msg):
        Labels(names=["a", "b"], values=np.array([[0, 0]], dtype=np.float64))


def test_position():
    labels = Labels(
        names=["a", "b"],
        values=np.array([[0, 0], [1, 0], [2, 2], [2, 3]]),
    )

    assert labels.position((0, 0)) == 0
    assert labels.position((2, 3)) == 3
    assert labels.position((2, -1)) is None

    assert (0, 0) in labels
    assert (2, 3) in labels
    assert (2, -1) not in labels


def test_not_writeable():
    labels = Labels(
        names=["a", "b"],
        values=np.array([[0, 0]]),
    )

    with pytest.raises(ValueError, match="assignment destination is read-only"):
        labels.values[0][0] = 4


def test_invalid_names():
    msg = "invalid parameter: 'not an ident' is not a valid label name"
    with pytest.raises(EquistoreError, match=msg):
        _ = Labels(
            names=["not an ident"],
            values=np.array([[0]]),
        )


def test_custom_constructors():
    # Labels.single
    label = Labels.single()
    assert label.names == ["_"]
    assert label.values.shape == (1, 1)

    # Labels.range
    labels = Labels.range("name", 10)
    assert labels.values.shape == (10, 1)
    assert labels.names == ["name"]
    np.testing.assert_equal(labels.values.reshape((-1,)), np.arange(10))

    message = "Labels names must be strings, got <class 'int'> instead"
    with pytest.raises(TypeError, match=message):
        Labels.range(0, 1)


def test_view():
    labels = Labels(names=("aaa", "bbb"), values=np.array([[1, 2], [3, 4]]))

    assert not labels.is_view()

    view = labels["aaa"]
    assert view.is_view()
    assert view.names == ["aaa"]
    np.testing.assert_equal(view.values, np.array([[1], [3]]))

    view = labels["bbb"]
    assert view.names == ["bbb"]
    np.testing.assert_equal(view.values, np.array([[2], [4]]))

    view = labels[["bbb"]]
    assert view.is_view()
    assert view.names == ["bbb"]
    np.testing.assert_equal(view.values, np.array([[2], [4]]))

    view = labels[["bbb", "aaa"]]
    assert view.is_view()
    assert view.names == ["bbb", "aaa"]
    np.testing.assert_equal(view.values, np.array([[2, 1], [4, 3]]))

    view = labels[["aaa", "aaa", "aaa"]]
    assert view.names == ["aaa", "aaa", "aaa"]
    np.testing.assert_equal(view.values, np.array([[1, 1, 1], [3, 3, 3]]))

    message = "'ccc' not found in the dimensions of these Labels"
    with pytest.raises(ValueError, match=message):
        labels["ccc"]

    message = "Labels names must be strings, got <class 'int'> instead"
    with pytest.raises(TypeError, match=message):
        labels[1, 2]

    view = labels["aaa"]
    message = "can not call `position` on a Labels view, call `to_owned` before"
    with pytest.raises(ValueError, match=message):
        view.position([1])

    owned = view.to_owned()
    assert not owned.is_view()
    assert owned.position([1]) == 0
    assert owned.position([-1]) is None

    view = labels[["aaa", "aaa"]]
    message = "invalid parameter: labels names must be unique, got 'aaa' multiple times"
    with pytest.raises(EquistoreError, match=message):
        view.to_owned()


def test_repr():
    labels = Labels(names=("aaa", "bbb"), values=np.array([[1, 2], [3, 4]]))

    expected = "Labels(\n    aaa  bbb\n     1    2\n     3    4\n)"
    assert str(labels) == expected
    assert repr(labels) == expected

    expected = "LabelsEntry(aaa=1, bbb=2)"
    assert str(labels[0]) == expected
    assert repr(labels[0]) == expected

    labels = Labels(
        names=("aaa", "bbb"),
        values=np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]),
    )

    expected = """ aaa  bbb
  0    0
  1    1
  2    2
  3    3
  4    4
  5    5
  6    6"""
    assert labels.print(max_entries=-1) == expected

    expected = """ aaa  bbb
  0    0
   ...
  6    6"""
    assert labels.print(max_entries=0) == expected
    assert labels.print(max_entries=1) == expected
    assert labels.print(max_entries=2) == expected

    expected = """ aaa  bbb
  0    0
  1    1
  2    2
   ...
  5    5
  6    6"""
    assert labels.print(max_entries=5) == expected

    expected = """ aaa  bbb
     0    0
     1    1
      ...
     6    6"""
    assert labels.print(max_entries=3, indent=3) == expected

    labels = Labels(names=("aaa", "bbb"), values=np.array([[0, 0], [0, 1]]))
    expected = "LabelsView(\n    bbb\n     0\n     1\n)"
    assert str(labels["bbb"]) == expected

    labels = Labels(
        names=("aaa", "bbb"), values=np.array([[111111111, 2], [3, 444444444]])
    )

    expected = """Labels(
       aaa        bbb
    111111111      2
        3      444444444
)"""
    assert str(labels) == expected


def test_indexing():
    labels = Labels(names=("a", "b"), values=np.array([[1, 2], [3, 4]]))

    # indexing labels with integer
    entry = labels[0]
    assert entry.names == ["a", "b"]
    np.testing.assert_equal(entry.values, np.array([1, 2]))

    entry = labels[-1]
    assert entry.names == ["a", "b"]
    np.testing.assert_equal(entry.values, np.array([3, 4]))

    # indexing labels errors
    message = "index 3 is out of bounds for axis 0 with size 2"
    with pytest.raises(IndexError, match=message):
        labels[3]

    message = "index -7 is out of bounds for axis 0 with size 2"
    with pytest.raises(IndexError, match=message):
        labels[-7]

    # indexing entry with integer
    entry = labels[0]
    assert entry[0] == 1
    assert entry[1] == 2
    assert tuple(entry) == (1, 2)

    # indexing entry with string
    assert entry["a"] == 1
    assert entry["b"] == 2

    # indexing entry errors
    message = "can only index LabelsEntry with str or int, got <class 'tuple'>"
    with pytest.raises(TypeError, match=message):
        entry[1, 2]

    message = "can only index LabelsEntry with str or int, got <class 'list'>"
    with pytest.raises(TypeError, match=message):
        entry[["a", "b"]]


def test_iter():
    labels = Labels(names=("a", "b"), values=np.array([[0, 0], [0, 1]]))

    for i, values in enumerate(labels[1]):
        assert values == i

    for i, entry in enumerate(labels):
        assert tuple(entry) == (0, i)

    # expand labels entry to tuples during iteration
    for i, (a, b) in enumerate(labels):
        assert a == 0
        assert b == i


def test_eq():
    labels_1 = Labels(names=("a", "b"), values=np.array([[0, 0], [0, 1]]))
    labels_2 = Labels(names=("a", "b"), values=np.array([[0, 0], [0, 1]]))
    labels_3 = Labels(names=("a", "b"), values=np.array([[0, 2], [0, 1]]))
    labels_4 = Labels(names=("a", "c"), values=np.array([[0, 0], [0, 1]]))

    # Labels equality
    assert labels_1 == labels_2
    assert labels_1 != labels_3
    assert labels_1 != labels_4

    # LabelsEntry equality
    assert labels_1[0] == labels_1[0]
    assert labels_1[0] == labels_2[0]
    assert labels_1[0] != labels_3[0]
    assert labels_1[0] != labels_4[0]
    assert labels_1[1] == labels_3[1]


def test_union():
    first = Labels(["aa", "bb"], np.array([[0, 1], [1, 2]]))
    second = Labels(["aa", "bb"], np.array([[2, 3], [1, 2], [4, 5]]))

    union = first.union(second)
    assert union.names == ["aa", "bb"]
    assert np.all(union.values == np.array([[0, 1], [1, 2], [2, 3], [4, 5]]))

    union_2, first_mapping, second_mapping = first.union_and_mapping(second)

    assert union == union_2
    assert np.all(first_mapping == np.array([0, 1]))
    assert np.all(second_mapping == np.array([2, 1, 3]))


def test_intersection():
    first = Labels(["aa", "bb"], np.array([[0, 1], [1, 2]]))
    second = Labels(["aa", "bb"], np.array([[2, 3], [1, 2], [4, 5]]))

    intersection = first.intersection(second)
    assert intersection.names == ["aa", "bb"]
    assert np.all(intersection.values == np.array([[1, 2]]))

    intersection_2, first_mapping, second_mapping = first.intersection_and_mapping(
        second
    )

    assert intersection == intersection_2
    assert np.all(first_mapping == np.array([-1, 0]))
    assert np.all(second_mapping == np.array([-1, 0, -1]))
