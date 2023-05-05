from typing import List, Optional

import pytest
import torch
from torch import Tensor

from equistore.torch import Labels


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


def test_constructor_errors():
    message = (
        "invalid Labels: the names must have an entry for each column of the array"
    )
    with pytest.raises(ValueError, match=message):
        _ = Labels(names="ab", values=torch.IntTensor([[0, 0], [0, 1]]))

    message = "names must be a tuple of strings"
    with pytest.raises(ValueError, match=message):
        _ = Labels(names=(3, 4), values=torch.IntTensor([[0, 0], [0, 1]]))

    message = "names must be a list of strings"
    with pytest.raises(ValueError, match=message):
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


def test_position():
    labels = Labels(names=("a", "b"), values=torch.IntTensor([[0, 0], [0, 1]]))

    assert labels.position(torch.IntTensor([0, 0])) == 0
    assert labels.position(torch.IntTensor([1, 0])) is None

    # list
    assert labels.position([0, 1]) == 1
    # tuple
    assert labels.position((0, 1)) == 1
    # # TODO: other Labels
    # assert labels.position(labels[0]) == 0

    message = (
        "parameter to Labels::positions must be a tensor or list/tuple of integers"
    )
    with pytest.raises(TypeError, match=message):
        _ = labels.position(3)


# define a wrapper class to make sure the types TorchScript uses for of all
# C-defined functions matches what we expect
class LabelsWrap:
    def __init__(self, names: List[str], values: Tensor):
        self._c = Labels(names, values)

    def __len__(self) -> int:
        return self._c.__len__()

    def names(self) -> List[str]:
        return self._c.names

    def values(self) -> Tensor:
        return self._c.values

    def position(self, entry: List[int]) -> Optional[int]:
        return self._c.position(entry=entry)


def test_script():
    class TestModule(torch.nn.Module):
        def forward(self, x: LabelsWrap) -> LabelsWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)
