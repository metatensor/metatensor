import os
import zipfile
from typing import Dict, List, Optional

import pytest
import torch

from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
    check_atomistic_model,
)


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
                samples=Labels("s", torch.tensor([[0]])),
                components=torch.jit.annotate(List[Labels], []),
                properties=Labels("p", torch.tensor([[0]])),
            )
            tensor = TensorMap(Labels("_", torch.tensor([[0]])), [block])
            return {
                "dummy": tensor,
            }
        else:
            return {}

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [
            NeighborListOptions(cutoff=1.2, full_list=False),
            NeighborListOptions(cutoff=4.3, full_list=True),
            NeighborListOptions(cutoff=1.2, full_list=False),
        ]


@pytest.fixture
def model():
    model = MinimalModel()
    model.train(False)

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1, 2, 3],
        interaction_range=4.3,
        outputs={
            "tests::dummy::long": ModelOutput(
                quantity="",
                unit="",
                per_atom=False,
                explicit_gradients=[],
            ),
        },
        supported_devices=["cpu"],
        dtype="float64",
    )

    metadata = ModelMetadata()
    return MetatensorAtomisticModel(model, metadata, capabilities)


def test_save(model, tmp_path):
    os.chdir(tmp_path)
    model.save("export.pt")

    with zipfile.ZipFile("export.pt") as file:
        assert "export/extra/metatensor-version" in file.namelist()
        assert "export/extra/torch-version" in file.namelist()

    check_atomistic_model("export.pt")


def test_export(model, tmp_path):
    os.chdir(tmp_path)
    match = r"`export\(\)` is deprecated, use `save\(\)` instead"
    with pytest.warns(DeprecationWarning, match=match):
        model.export("export.pt")


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

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [NeighborListOptions(1.0, False, self._name)]


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

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [NeighborListOptions(2.0, True, "other module")]


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


class EnergyEnsembleModel(torch.nn.Module):
    """A metatensor atomistic model returning an energy ensemble"""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        assert "energy_ensemble" in outputs
        assert not outputs["energy_ensemble"].per_atom
        assert selected_atoms is None

        return_dict: Dict[str, TensorMap] = {}
        block = TensorBlock(
            values=torch.tensor([[0.0, 1.0, 2.0]] * len(systems), dtype=torch.float64),
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels("ensemble_member", torch.tensor([[0], [1], [2]])),
        )
        return_dict["energy_ensemble"] = TensorMap(
            Labels("_", torch.tensor([[0]])), [block]
        )
        return return_dict


def test_requested_neighbor_lists():
    model = FullModel()
    model.train(False)

    capabilities = ModelCapabilities(
        interaction_range=0.0,
        supported_devices=["cpu"],
        dtype="float64",
    )
    atomistic = MetatensorAtomisticModel(model, ModelMetadata(), capabilities)
    requests = atomistic.requested_neighbor_lists()

    assert len(requests) == 2

    assert requests[0].cutoff == 1.0
    assert not requests[0].full_list
    assert requests[0].requestors() == [
        "first module",
        "FullModel.first",
        "second module",
        "FullModel.second",
    ]

    assert requests[1].cutoff == 2.0
    assert requests[1].full_list
    assert requests[1].requestors() == [
        "other module",
        "FullModel.other",
    ]


def test_bad_capabilities():
    model = FullModel()
    model.train(False)

    capabilities = ModelCapabilities(
        supported_devices=["cpu"],
        dtype="float64",
    )
    message = (
        "`capabilities.interaction_range` was not set, "
        "but it is required to run simulations"
    )
    with pytest.raises(ValueError, match=message):
        MetatensorAtomisticModel(model, ModelMetadata(), capabilities)

    capabilities = ModelCapabilities(
        interaction_range=12,
        dtype="float64",
    )
    message = (
        "`capabilities.supported_devices` was not set, "
        "but it is required to run simulations"
    )
    with pytest.raises(ValueError, match=message):
        MetatensorAtomisticModel(model, ModelMetadata(), capabilities)

    capabilities = ModelCapabilities(
        interaction_range=float("nan"),
        supported_devices=["cpu"],
        dtype="float64",
    )
    message = (
        "`capabilities.interaction_range` should be a float between 0 and infinity"
    )
    with pytest.raises(ValueError, match=message):
        MetatensorAtomisticModel(model, ModelMetadata(), capabilities)

    capabilities = ModelCapabilities(
        interaction_range=12.0,
        supported_devices=["cpu"],
    )
    message = "`capabilities.dtype` was not set, but it is required to run simulations"
    with pytest.raises(ValueError, match=message):
        MetatensorAtomisticModel(model, ModelMetadata(), capabilities)

    message = (
        "Invalid name for model output: 'not-a-standard'. "
        "Non-standard names should have the form '<domain>::<output>'."
    )
    with pytest.raises(ValueError, match=message):
        ModelCapabilities(outputs={"not-a-standard": ModelOutput()})

    message = (
        "Invalid name for model output: '::not-a-standard'. "
        "Non-standard names should have the form '<domain>::<output>'."
    )
    with pytest.raises(ValueError, match=message):
        ModelCapabilities(outputs={"::not-a-standard": ModelOutput()})

    message = (
        "Invalid name for model output: 'not-a-standard::'. "
        "Non-standard names should have the form '<domain>::<output>'."
    )
    with pytest.raises(ValueError, match=message):
        ModelCapabilities(outputs={"not-a-standard::": ModelOutput()})


def test_energy_ensemble_model():
    model = EnergyEnsembleModel()
    model.eval()

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1, 2, 3],
        interaction_range=4.3,
        outputs={
            "energy_ensemble": ModelOutput(
                quantity="",
                unit="",
                per_atom=False,
                explicit_gradients=[],
            ),
        },
        supported_devices=["cpu"],
        dtype="float64",
    )

    metadata = ModelMetadata()
    atomistic = MetatensorAtomisticModel(model, metadata, capabilities)

    system = System(
        types=torch.tensor([1, 2]),
        positions=torch.tensor(
            [[1, 1, 1], [0, 0, 0]], dtype=torch.float64, requires_grad=True
        ),
        cell=torch.zeros([3, 3], dtype=torch.float64),
    )

    outputs = {
        "energy_ensemble": ModelOutput(quantity="energy", unit="eV", per_atom=False),
    }
    options = ModelEvaluationOptions(
        length_unit="angstrom", outputs=outputs, selected_atoms=None
    )

    result = atomistic([system, system], options, check_consistency=True)
    assert "energy_ensemble" in result
    assert result["energy_ensemble"].keys == Labels("_", torch.tensor([[0]]))
    assert result["energy_ensemble"].block().values.shape[0] == 2
    assert result["energy_ensemble"].block().samples.names == ["system"]
    assert result["energy_ensemble"].block().properties.names == ["ensemble_member"]
