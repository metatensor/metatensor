import os
import zipfile
from typing import Dict, List, Optional

import pytest
import torch

from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
    check_atomistic_model,
    is_atomistic_model,
    load_atomistic_model,
    read_model_metadata,
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
            NeighborListOptions(cutoff=1.2, full_list=False, strict=True),
            NeighborListOptions(cutoff=4.3, full_list=True, strict=True),
            NeighborListOptions(cutoff=1.2, full_list=False, strict=False),
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
                sample_kind=["system"],
                explicit_gradients=[],
            ),
        },
        supported_devices=["cpu"],
        dtype="float64",
    )

    metadata = ModelMetadata()
    return MetatensorAtomisticModel(model, metadata, capabilities)


@pytest.fixture
def model_energy_nounit():
    model_energy_nounit = MinimalModel()
    model_energy_nounit.train(False)

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1, 2, 3],
        interaction_range=4.3,
        outputs={
            "energy": ModelOutput(
                quantity="",
                unit="",
                sample_kind=["system"],
                explicit_gradients=[],
            ),
        },
        supported_devices=["cpu"],
        dtype="float64",
    )

    metadata = ModelMetadata()
    return MetatensorAtomisticModel(model_energy_nounit, metadata, capabilities)


def test_save(model, tmp_path):
    os.chdir(tmp_path)
    model.save("export.pt")

    with zipfile.ZipFile("export.pt") as file:
        assert "export/extra/metatensor-version" in file.namelist()
        assert "export/extra/torch-version" in file.namelist()

    check_atomistic_model("export.pt")


def test_recreate(model, tmp_path):
    os.chdir(tmp_path)
    model.save("export.pt")
    model_loaded = load_atomistic_model("export.pt")
    model_loaded.save("export_new.pt")

    with zipfile.ZipFile("export_new.pt") as file:
        assert "export_new/extra/metatensor-version" in file.namelist()
        assert "export_new/extra/torch-version" in file.namelist()

    check_atomistic_model("export_new.pt")


def test_training_mode():
    model = MinimalModel()
    model.train(True)
    capabilities = ModelCapabilities(supported_devices=["cpu"], dtype="float64")

    with pytest.raises(ValueError, match="module should not be in training mode"):
        MetatensorAtomisticModel(model, ModelMetadata(), capabilities)


def test_save_warning_length_unit(model):
    model._capabilities.length_unit = ""
    match = r"No length unit was provided for the model."
    with pytest.warns(UserWarning, match=match):
        model.save("export.pt")


def test_save_warning_quantity(model_energy_nounit):
    match = r"No units were provided for output energy."
    with pytest.warns(UserWarning, match=match):
        model_energy_nounit.save("export.pt")


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
        return [NeighborListOptions(1.0, False, True, self._name)]


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
        return [NeighborListOptions(2.0, True, False, "other module")]


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


def test_requested_neighbor_lists(tmpdir):
    model = FullModel()
    model.train(False)

    capabilities = ModelCapabilities(
        interaction_range=0.0,
        length_unit="A",
        supported_devices=["cpu"],
        dtype="float64",
    )
    atomistic = MetatensorAtomisticModel(model, ModelMetadata(), capabilities)
    requests = atomistic.requested_neighbor_lists()

    assert len(requests) == 2

    assert requests[0].cutoff == 1.0
    assert not requests[0].full_list
    assert requests[0].strict
    assert requests[0].requestors() == [
        "first module",
        "FullModel.first",
        "second module",
        "FullModel.second",
    ]

    assert requests[1].cutoff == 2.0
    assert requests[1].full_list
    assert not requests[1].strict
    assert requests[1].requestors() == [
        "other module",
        "FullModel.other",
    ]

    # check these are still around after serialization/reload
    atomistic.save(os.path.join(tmpdir, "model.pt"))
    loaded = torch.jit.load(os.path.join(tmpdir, "model.pt"))
    requests = loaded.requested_neighbor_lists()

    assert len(requests) == 2

    assert requests[0].cutoff == 1.0
    assert not requests[0].full_list
    assert requests[0].strict
    assert requests[0].requestors() == [
        "first module",
        "FullModel.first",
        "second module",
        "FullModel.second",
    ]

    assert requests[1].cutoff == 2.0
    assert requests[1].full_list
    assert not requests[1].strict
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


def test_access_module(tmpdir):
    model = FullModel()
    model.train(False)

    capabilities = ModelCapabilities(
        length_unit="nm",
        interaction_range=0.0,
        supported_devices=["cpu"],
        dtype="float64",
    )
    atomistic = MetatensorAtomisticModel(model, ModelMetadata(), capabilities)

    # Access wrapped module
    assert atomistic.module is model

    atomistic.save(tmpdir / "export.pt")
    loaded_atomistic = load_atomistic_model(tmpdir / "export.pt")

    # Access wrapped module after loading
    loaded_atomistic.module

    # Verfify that it contains the original submodules
    loaded_atomistic.module.first
    loaded_atomistic.module.second
    loaded_atomistic.module.other


def test_is_atomistic_model(tmpdir):
    model = FullModel()
    model.train(False)

    capabilities = ModelCapabilities(
        length_unit="A",
        interaction_range=0.0,
        supported_devices=["cpu"],
        dtype="float64",
    )
    atomistic = MetatensorAtomisticModel(model, ModelMetadata(), capabilities)
    atomistic.save(tmpdir / "model.pt")

    scripted_atomistic = torch.jit.script(atomistic)
    loaded_atomistic = load_atomistic_model(tmpdir / "model.pt")

    assert is_atomistic_model(atomistic)
    assert is_atomistic_model(scripted_atomistic)
    assert is_atomistic_model(loaded_atomistic)

    match = "`module` should be a torch.nn.Module, not float"
    with pytest.raises(TypeError, match=match):
        is_atomistic_model(1.0)


def test_read_metadata(tmpdir):
    model = FullModel()
    model.train(False)

    capabilities = ModelCapabilities(
        length_unit="nm",
        interaction_range=0.0,
        supported_devices=["cpu"],
        dtype="float64",
    )
    metadata = ModelMetadata(
        name="NEW_SOTA",
        description="A SOTA model",
        authors=["Alice", "Bob"],
        references={"implementation": ["doi:1234", "arXiv:1234"]},
    )
    atomistic = MetatensorAtomisticModel(model, metadata, capabilities)
    atomistic.save(tmpdir / "model.pt")

    extracted_metadata = read_model_metadata(str(tmpdir / "model.pt"))

    assert str(extracted_metadata) == str(metadata)
