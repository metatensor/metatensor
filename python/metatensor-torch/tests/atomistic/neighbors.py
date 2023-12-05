from typing import Dict, List, Optional

import torch
from packaging import version

from metatensor.torch import TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelOutput,
    NeighborsListOptions,
    System,
)


def test_neighbors_lists_options():
    options = NeighborsListOptions(3.4, True, "hello")

    assert options.model_cutoff == 3.4
    assert options.full_list
    assert options.requestors() == ["hello"]

    options.add_requestor("another one")
    assert options.requestors() == ["hello", "another one"]

    # No empty requestors, no duplicated requestors
    options.add_requestor("")
    options.add_requestor("hello")
    assert options.requestors() == ["hello", "another one"]

    assert NeighborsListOptions(3.4, True, "a") == NeighborsListOptions(3.4, True, "b")
    assert NeighborsListOptions(3.4, True) != NeighborsListOptions(3.4, False)
    assert NeighborsListOptions(3.4, True) != NeighborsListOptions(3.5, True)

    expected = "NeighborsListOptions(cutoff=3.400000, full_list=True)"
    assert str(options) == expected

    expected = """NeighborsListOptions
    model_cutoff: 3.400000
    full_list: True
    requested by:
        - hello
        - another one
"""
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert repr(options) == expected


class ExampleModule(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self._name = name

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[List[List[int]]],
    ) -> Dict[str, TensorMap]:
        return {}

    def requested_neighbors_lists(self) -> List[NeighborsListOptions]:
        return [NeighborsListOptions(1.0, False, self._name)]


class OtherModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[List[List[int]]],
    ) -> Dict[str, TensorMap]:
        return {}

    def requested_neighbors_lists(self) -> List[NeighborsListOptions]:
        return [NeighborsListOptions(2.0, True, "other module")]


class FullModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = ExampleModule("first module")
        self.second = ExampleModule("second module")
        self.other = OtherModule()

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[List[List[int]]],
    ) -> Dict[str, TensorMap]:
        result = self.first(systems, outputs, selected_atoms)
        result.update(self.second(systems, outputs, selected_atoms))
        result.update(self.other(systems, outputs, selected_atoms))

        return result


def test_requested_neighbors_lists():
    model = FullModel()
    model.train(False)

    atomistic = MetatensorAtomisticModel(model, ModelCapabilities())
    requests = atomistic.requested_neighbors_lists()

    assert len(requests) == 2

    assert requests[0].model_cutoff == 1.0
    assert not requests[0].full_list
    assert requests[0].requestors() == [
        "first module",
        "FullModel.first",
        "second module",
        "FullModel.second",
    ]

    assert requests[1].model_cutoff == 2.0
    assert requests[1].full_list
    assert requests[1].requestors() == [
        "other module",
        "FullModel.other",
    ]
