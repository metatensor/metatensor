import zipfile
from typing import Dict, List

import pytest
import torch

from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import (
    MetatensorAtomisticModule,
    ModelCapabilities,
    ModelOutput,
    ModelRunOptions,
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
        self, system: System, run_options: ModelRunOptions
    ) -> Dict[str, TensorBlock]:
        if "dummy" in run_options.outputs:
            return {
                "dummy": TensorBlock(
                    values=torch.tensor([[0.0]]),
                    samples=Labels("s", torch.IntTensor([[0]])),
                    components=[],
                    properties=Labels("p", torch.IntTensor([[0]])),
                )
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
                forward_gradients=[],
            ),
        },
    )

    return MetatensorAtomisticModule(model, capabilities)


def test_export(model, tmpdir):
    with tmpdir.as_cwd():
        model.export("export.pt")

        with zipfile.ZipFile("export.pt") as file:
            assert "export/extra/metatensor-version" in file.namelist()
            assert "export/extra/torch-version" in file.namelist()

        check_atomistic_model("export.pt")
