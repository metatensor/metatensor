import zipfile
from typing import Dict, List, Optional

import pytest
import torch

from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelOutput,
    NeighborsListOptions,
    System,
    check_atomistic_model,
)
from metatensor.torch.atomistic.units import KNOWN_QUANTITIES


try:
    import ase.units

    HAVE_ASE = True
except ImportError:
    HAVE_ASE = False


class MinimalModel(torch.nn.Module):
    """The simplest possible metatensor atomistic model"""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if "dummy" in outputs:
            block = TensorBlock(
                values=torch.tensor([[0.0]]),
                samples=Labels("s", torch.IntTensor([[0]])),
                components=[],
                properties=Labels("p", torch.IntTensor([[0]])),
            )
            return {
                "dummy": TensorMap(Labels("_", torch.IntTensor([[0]])), [block]),
            }
        else:
            return {}

    def requested_neighbors_lists(self) -> List[NeighborsListOptions]:
        return [
            NeighborsListOptions(model_cutoff=1.2, full_list=False),
            NeighborsListOptions(model_cutoff=4.3, full_list=True),
            NeighborsListOptions(model_cutoff=1.2, full_list=False),
        ]


@pytest.fixture
def model():
    model = MinimalModel()
    model.train(False)

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        species=[1, 2, 3],
        outputs={
            "dummy": ModelOutput(
                quantity="",
                unit="",
                per_atom=False,
                explicit_gradients=[],
            ),
        },
    )

    return MetatensorAtomisticModel(model, capabilities)


def test_export(model, tmpdir):
    with tmpdir.as_cwd():
        model.export("export.pt")

        with zipfile.ZipFile("export.pt") as file:
            assert "export/extra/metatensor-version" in file.namelist()
            assert "export/extra/torch-version" in file.namelist()

        check_atomistic_model("export.pt")


class ExampleModule(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self._name = name

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
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
        selected_atoms: Optional[Labels],
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
        selected_atoms: Optional[Labels],
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


@pytest.mark.skipif(not HAVE_ASE, reason="this tests requires ASE units")
def test_units():
    length = KNOWN_QUANTITIES["length"]
    assert length._baseline == "angstrom"
    for name, value in length._conversions.items():
        if name == "angstrom":
            assert value == ase.units.Ang / ase.units.Ang
        elif name == "bohr":
            assert value == ase.units.Bohr / ase.units.Ang
        elif name in ["nanometer", "nm"]:
            assert value == ase.units.nm / ase.units.Ang
        else:
            raise Exception(f"missing name in test: {name}")

    energy = KNOWN_QUANTITIES["energy"]
    assert energy._baseline == "ev"
    for name, value in energy._conversions.items():
        if name == "ev":
            assert value == ase.units.eV / ase.units.eV
        elif name == "mev":
            assert value == (ase.units.eV / 1000) / ase.units.eV
        elif name == "hartree":
            assert value == ase.units.Hartree / ase.units.eV
        elif name == "kcal/mol":
            assert value == (ase.units.kcal / ase.units.mol) / ase.units.eV
        elif name == "kj/mol":
            assert value == (ase.units.kJ / ase.units.mol) / ase.units.eV
        else:
            raise Exception(f"missing name in test: {name}")
