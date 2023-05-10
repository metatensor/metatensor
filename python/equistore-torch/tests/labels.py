from typing import Any, List, Optional, Union

import pytest
import torch
from torch import Tensor

from equistore.torch import Labels, LabelsEntry


def test_constructor():
    # keyword arguments + name is a tuple
    labels = Labels(names=("a", "b"), values=torch.IntTensor([[0, 0], [0, 1]]))

    assert len(labels) == 2

    assert labels.names == ["a", "b"]
    assert torch.all(labels.values == torch.Tensor([[0, 0], [0, 1]]))

    # positional arguments + names is a list
    labels = Labels(["c"], torch.LongTensor([[0], [2], [-1]]))
    assert labels.names == ["c"]
    assert torch.all(labels.values == torch.Tensor([[0], [2], [-1]]))

    # single string for names
    labels = Labels(names="test", values=torch.IntTensor([[0], [1]]))

    assert labels.names == ["test"]
    assert torch.all(labels.values == torch.Tensor([[0], [1]]))

    labels = Labels.single()
    assert labels.names == ["_"]
    assert torch.all(labels.values == torch.Tensor([[0]]))

    labels = Labels.empty(["a", "bb", "ccc"])
    assert labels.names == ["a", "bb", "ccc"]
    assert labels.values.shape == (0, 3)

    labels = Labels.range("test", 33)
    assert labels.names == ["test"]
    assert torch.all(
        labels.values == torch.arange(33, dtype=torch.int32).reshape(33, 1)
    )


def test_constructor_errors():
    message = (
        "invalid Labels: the names must have an entry for each column of the array"
    )
    with pytest.raises(ValueError, match=message):
        _ = Labels(names="ab", values=torch.IntTensor([[0, 0], [0, 1]]))

    message = "names must be a tuple of strings, got element with type 'int' instead"
    with pytest.raises(TypeError, match=message):
        _ = Labels(names=(3, 4), values=torch.IntTensor([[0, 0], [0, 1]]))

    message = "names must be a list of strings, got element with type 'int' instead"
    with pytest.raises(TypeError, match=message):
        _ = Labels(names=[3, 4], values=torch.IntTensor([[0, 0], [0, 1]]))

    message = (
        "Expected a value of type 'Tensor' for argument 'values' but "
        + "instead found type 'tuple"
    )
    with pytest.raises(RuntimeError, match=message):
        _ = Labels(names="test", values=(3, 4))

    message = "Labels values must be an Tensor of 32-bit integers"
    with pytest.raises(ValueError, match=message):
        _ = Labels(names="test", values=torch.Tensor([[0, 0], [0, 1]]))

    message = "Labels values must be a 2D Tensor"
    with pytest.raises(ValueError, match=message):
        _ = Labels(names="test", values=torch.IntTensor([0, 1]))

    # check that normal validation from equistore_core is passed through with
    # exceptions
    message = "invalid parameter: 'not an ident' is not a valid label name"
    with pytest.raises(RuntimeError, match=message):
        _ = Labels(names="not an ident", values=torch.IntTensor([[0]]))


def test_view():
    labels = Labels(names=("aaa", "bbb"), values=torch.IntTensor([[1, 2], [3, 4]]))

    assert not labels.is_view()

    view = labels["aaa"]
    assert view.is_view()
    assert view.names == ["aaa"]
    assert torch.all(view.values == torch.IntTensor([[1], [3]]))

    view = labels["bbb"]
    assert view.names == ["bbb"]
    assert torch.all(view.values == torch.IntTensor([[2], [4]]))

    view = labels[["bbb"]]
    assert view.is_view()
    assert view.names == ["bbb"]
    assert torch.all(view.values == torch.IntTensor([[2], [4]]))

    view = labels[["bbb", "aaa"]]
    assert view.is_view()
    assert view.names == ["bbb", "aaa"]
    assert torch.all(view.values == torch.IntTensor([[2, 1], [4, 3]]))

    view = labels[["aaa", "aaa", "aaa"]]
    assert view.names == ["aaa", "aaa", "aaa"]
    assert torch.all(view.values == torch.IntTensor([[1, 1, 1], [3, 3, 3]]))

    message = "'ccc' not found in the dimensions of these Labels"
    with pytest.raises(ValueError, match=message):
        labels["ccc"]

    message = "index must be a tuple of strings, got element with type 'int' instead"
    with pytest.raises(TypeError, match=message):
        labels[1, 2]

    view = labels["aaa"]
    message = "can not call this function on Labels view, call to_owned first"
    with pytest.raises(ValueError, match=message):
        view.position([1])

    owned = view.to_owned()
    assert not owned.is_view()
    assert owned.position([1]) == 0
    assert owned.position([-1]) is None

    view = labels[["aaa", "aaa"]]
    message = "invalid parameter: labels names must be unique, got 'aaa' multiple times"
    with pytest.raises(RuntimeError, match=message):
        view.to_owned()


def test_repr():
    labels = Labels(names=("aaa", "bbb"), values=torch.IntTensor([[1, 2], [3, 4]]))

    expected = "Labels(\n    aaa  bbb\n     1    2\n     3    4\n)"
    assert str(labels) == expected
    # custom __repr__ definitions are not passed to Python, we need to wait for
    # https://github.com/pytorch/pytorch/pull/100724 to be merged and available
    # assert repr(labels) == expected

    expected = "LabelsEntry(aaa=1, bbb=2)"
    assert str(labels[0]) == expected
    # assert repr(labels[0]) == expected

    labels = Labels(
        names=("aaa", "bbb"),
        values=torch.IntTensor(
            [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
        ),
    )

    expected = """ aaa  bbb
  0    0
  1    1
  2    2
  3    3
  4    4
  5    5
  6    6
"""
    assert labels.print(max_entries=-1) == expected

    expected = """ aaa  bbb
  0    0
   ...
  6    6
"""
    assert labels.print(max_entries=0) == expected
    assert labels.print(max_entries=1) == expected
    assert labels.print(max_entries=2) == expected

    expected = """ aaa  bbb
  0    0
  1    1
  2    2
   ...
  5    5
  6    6
"""
    assert labels.print(max_entries=5) == expected

    expected = """ aaa  bbb
     0    0
     1    1
      ...
     6    6
"""
    assert labels.print(max_entries=3, indent=3) == expected

    labels = Labels(names=("aaa", "bbb"), values=torch.IntTensor([[0, 0], [0, 1]]))
    expected = "LabelsView(\n    bbb\n     0\n     1\n)"
    assert str(labels["bbb"]) == expected

    labels = Labels(
        names=("aaa", "bbb"), values=torch.IntTensor([[111111111, 2], [3, 444444444]])
    )

    expected = """Labels(
       aaa        bbb
    111111111      2
        3      444444444
)"""
    assert str(labels) == expected


def test_indexing():
    labels = Labels(names=("a", "b"), values=torch.IntTensor([[1, 2], [3, 4]]))

    # indexing labels with integer
    entry = labels[0]
    assert entry.names == ["a", "b"]
    assert torch.all(entry.values == torch.IntTensor([[1, 2]]))

    entry = labels[-1]
    assert entry.names == ["a", "b"]
    assert torch.all(entry.values == torch.IntTensor([[3, 4]]))

    # indexing labels errors
    message = "out of range for tensor of size \\[2, 2\\] at dimension 0"
    with pytest.raises(IndexError, match=message):
        labels[3]

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
    message = (
        "LabelsEntry can only be indexed by int or str, got '\\(int, int\\)' instead"
    )
    with pytest.raises(TypeError, match=message):
        entry[1, 2]

    message = "LabelsEntry can only be indexed by int or str, got 'str\\[\\]' instead"
    with pytest.raises(TypeError, match=message):
        entry[["a", "b"]]


def test_iter():
    # we can iterate over Labels/LabelsEntry since they define both __len__ and
    # __getitem__
    labels = Labels(names=("a", "b"), values=torch.IntTensor([[0, 0], [0, 1]]))

    for i, values in enumerate(labels[1]):
        assert values == i

    for i, entry in enumerate(labels):
        assert tuple(entry) == (0, i)

    # expand labels entry to tuples during iteration
    for i, (a, b) in enumerate(labels):
        assert a == 0
        assert b == i


def test_eq():
    labels_1 = Labels(names=("a", "b"), values=torch.IntTensor([[0, 0], [0, 1]]))
    labels_2 = Labels(names=("a", "b"), values=torch.IntTensor([[0, 0], [0, 1]]))
    labels_3 = Labels(names=("a", "b"), values=torch.IntTensor([[0, 2], [0, 1]]))
    labels_4 = Labels(names=("a", "c"), values=torch.IntTensor([[0, 0], [0, 1]]))

    assert labels_1 == labels_2
    assert labels_1 != labels_3
    assert labels_1 != labels_4

    assert labels_1[0] == labels_1[0]
    assert labels_1[0] == labels_2[0]
    assert labels_1[0] != labels_3[0]
    assert labels_1[0] != labels_4[0]
    assert labels_1[1] == labels_3[1]


def test_position():
    labels = Labels(names=("a", "b"), values=torch.IntTensor([[0, 0], [0, 1]]))

    assert labels.position(torch.IntTensor([0, 0])) == 0
    assert labels.position(torch.IntTensor([1, 0])) is None
    assert torch.IntTensor([0, 0]) in labels
    assert torch.IntTensor([1, 0]) not in labels

    # list
    assert labels.position([0, 1]) == 1
    assert labels.position([1, 0]) is None
    assert [0, 1] in labels
    assert [1, 0] not in labels

    # tuple
    assert labels.position((0, 1)) == 1
    assert labels.position((1, 0)) is None
    assert (0, 1) in labels
    assert (1, 0) not in labels

    # LabelsEntry
    assert labels.position(labels[0]) == 0
    assert labels.position(labels[1]) == 1

    other_labels = Labels(names=("a", "b"), values=torch.IntTensor([[0, 1], [2, 3]]))
    assert labels.position(other_labels[0]) == 1
    assert labels.position(other_labels[1]) is None

    # make sure nothing breaks when using the same column multiple time in a view
    view = other_labels[["a", "a"]]
    assert labels.position(view[0]) == 0

    message = (
        "parameter to Labels::positions must be a LabelsEntry, tensor, "
        "or list/tuple of integers"
    )
    with pytest.raises(TypeError, match=message):
        _ = labels.position(3)


# define a wrapper class to make sure the types TorchScript uses for of all
# C-defined functions matches what we expect
class LabelsWrap:
    def __init__(self, names: List[str], values: Tensor):
        self._c = Labels(names, values)

    def __len___(self) -> int:
        return self._c.__len__()

    def __str__(self) -> str:
        return self._c.__str__()

    def __repr__(self) -> str:
        return self._c.__repr__()

    def __eq__(self, other: Labels) -> bool:
        return self._c.__eq__(other=other)

    def __ne__(self, other: Labels) -> bool:
        return self._c.__ne__(other=other)

    # we can not set the return type to Union[Labels, LabelsEntry] from C++
    # so it is set to Any
    def __getitem__(self, index: Union[int, str, List[str]]) -> Any:
        return self._c.__getitem__(index=index)

    def contains(self, entry: Union[List[int], LabelsEntry]) -> bool:
        return self._c.__contains__(entry=entry)

    def names(self) -> List[str]:
        return self._c.names

    def values(self) -> Tensor:
        return self._c.values

    def position(self, entry: Union[List[int], LabelsEntry]) -> Optional[int]:
        return self._c.position(entry=entry)

    def print_(self, max_entries: int, indent: int) -> str:
        return self._c.print(max_entries=max_entries, indent=indent)

    def to_owned(self) -> Labels:
        return self._c.to_owned()

    def entry(self, index: int) -> LabelsEntry:
        return self._c.entry(index=index)

    def view(self, names: Union[str, List[str]]) -> Labels:
        return self._c.view(names=names)

    def single(self) -> Labels:
        return Labels.single()

    def empty(self, names: Union[str, List[str]]) -> Labels:
        return Labels.empty(names)

    def range_(self, name: str, end: int) -> Labels:
        return Labels.range(name, end)


class LabelsEntryWrap:
    def __init__(self):
        labels = Labels(["a"], torch.tensor([]))
        entry = labels[0]
        assert isinstance(entry, LabelsEntry)
        self._c = entry

    def __str__(self) -> str:
        return self._c.__str__()

    def __repr__(self) -> str:
        return self._c.__repr__()

    def __len__(self) -> int:
        return self._c.__len__()

    def __getitem__(self, index: Union[int, str]) -> int:
        return self._c.__getitem__(index=index)

    def __eq__(self, other: LabelsEntry) -> bool:
        return self._c.__eq__(other=other)

    def __ne__(self, other: LabelsEntry) -> bool:
        return self._c.__ne__(other=other)

    def names(self) -> List[str]:
        return self._c.names

    def values(self) -> Tensor:
        return self._c.values


def test_script():
    class TestModule(torch.nn.Module):
        def forward(self, x: LabelsWrap) -> LabelsWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)

    class TestModuleEntry(torch.nn.Module):
        def forward(self, x: LabelsEntryWrap) -> LabelsEntryWrap:
            return x

    module = TestModuleEntry()
    module = torch.jit.script(module)
