from typing import Dict, List, Optional

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
            properties=Labels("energy", torch.tensor([[0], [1], [2]])),
        )
        return_dict["energy_ensemble"] = TensorMap(
            Labels("_", torch.tensor([[0]])), [block]
        )
        return return_dict


def test_energy_ensemble_model():
    model = EnergyEnsembleModel()

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1, 2, 3],
        interaction_range=4.3,
        outputs={"energy_ensemble": ModelOutput(per_atom=False)},
        supported_devices=["cpu"],
        dtype="float64",
    )

    atomistic = MetatensorAtomisticModel(model.eval(), ModelMetadata(), capabilities)

    system = System(
        types=torch.tensor([1, 2, 3]),
        positions=torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=torch.float64),
        cell=torch.zeros([3, 3], dtype=torch.float64),
    )

    options = ModelEvaluationOptions(
        outputs={"energy_ensemble": ModelOutput(per_atom=False)}
    )

    result = atomistic([system, system], options, check_consistency=True)
    assert "energy_ensemble" in result

    ensemble = result["energy_ensemble"]

    assert ensemble.keys == Labels("_", torch.tensor([[0]]))
    assert list(ensemble.block().values.shape) == [2, 3]
    assert ensemble.block().samples.names == ["system"]
    assert ensemble.block().properties.names == ["energy"]
