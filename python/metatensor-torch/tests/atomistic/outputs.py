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
    System,
)


@pytest.fixture
def system():
    return System(
        types=torch.tensor([1, 2, 3]),
        positions=torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=torch.float64),
        cell=torch.zeros([3, 3], dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


@pytest.fixture
def get_capabilities() -> callable:
    def _create_capabilities(output_name: str) -> ModelCapabilities:
        return ModelCapabilities(
            length_unit="angstrom",
            atomic_types=[1, 2, 3],
            interaction_range=4.3,
            outputs={output_name: ModelOutput(sample_kind=["system"])},
            supported_devices=["cpu"],
            dtype="float64",
        )

    return _create_capabilities


class BaseAtomisticModel(torch.nn.Module):
    """Base class for metatensor atomistic models"""

    def __init__(self, output_name: str):
        super().__init__()
        self.output_name = output_name

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        assert self.output_name in outputs
        assert outputs[self.output_name].sample_kind == ["system"]
        assert selected_atoms is None

        block = TensorBlock(
            values=torch.tensor([[0.0, 1.0, 2.0]] * len(systems), dtype=torch.float64),
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels("energy", torch.tensor([[0], [1], [2]])),
        )
        return {self.output_name: TensorMap(Labels("_", torch.tensor([[0]])), [block])}


class EnergyEnsembleModel(BaseAtomisticModel):
    """A metatensor atomistic model returning an energy ensemble"""

    def __init__(self):
        super().__init__("energy_ensemble")


class FeaturesModel(BaseAtomisticModel):
    """A metatensor atomistic model returning features"""

    def __init__(self):
        super().__init__("features")


def test_energy_ensemble_model(system, get_capabilities):
    model = EnergyEnsembleModel()
    capabilities = get_capabilities("energy_ensemble")
    atomistic = MetatensorAtomisticModel(model.eval(), ModelMetadata(), capabilities)

    options = ModelEvaluationOptions(
        outputs={"energy_ensemble": ModelOutput(sample_kind=["system"])}
    )

    result = atomistic([system, system], options, check_consistency=True)
    assert "energy_ensemble" in result

    ensemble = result["energy_ensemble"]

    assert ensemble.keys == Labels("_", torch.tensor([[0]]))
    assert list(ensemble.block().values.shape) == [2, 3]
    assert ensemble.block().samples.names == ["system"]
    assert ensemble.block().properties.names == ["energy"]


def test_features_model(system, get_capabilities):
    model = FeaturesModel()
    capabilities = get_capabilities("features")
    atomistic = MetatensorAtomisticModel(model.eval(), ModelMetadata(), capabilities)

    options = ModelEvaluationOptions(
        outputs={"features": ModelOutput(sample_kind=["system"])}
    )

    result = atomistic([system, system], options, check_consistency=True)
    assert "features" in result

    features = result["features"]
    assert features.keys == Labels("_", torch.tensor([[0]]))
    assert list(features.block().values.shape) == [2, 3]
    assert features.block().samples.names == ["system"]
    assert features.block().properties.names == ["energy"]
    assert features.block().components == []
    assert len(result["features"].blocks()) == 1
