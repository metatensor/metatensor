import os
from typing import Dict, List

import numpy as np
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
)
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator


ase = pytest.importorskip("ase")
import ase.build  # noqa  isort: skip
import ase.calculators.lj  # noqa isort: skip


class LennardJones(torch.nn.Module):
    """
    Implementation of Lennard-Jones potential using the ``MetatensorAtomisticModule``
    API. This is then compared against the implementation inside ASE.
    """

    def __init__(self, cutoff, params):
        super().__init__()
        self._nl_options = NeighborsListOptions(model_cutoff=cutoff, full_list=False)

        self._lj_params = {}
        for (a, b), (s, e) in params.items():
            # shift the energy to 0 at the cutoff
            shift = 4 * e * ((s / cutoff) ** 12 - (s / cutoff) ** 6)
            if a in self._lj_params:
                self._lj_params[a][b] = (s, e, shift)
            else:
                self._lj_params[a] = {b: (s, e, shift)}

    def forward(
        self, system: System, run_options: ModelRunOptions
    ) -> Dict[str, TensorBlock]:
        if "energy" not in run_options.outputs:
            return {}

        if run_options.selected_atoms is not None:
            # TODO: add selected_atoms
            raise NotImplementedError("selected atoms is not yet implemented")

        assert torch.all(
            system.positions.samples.column("atom")
            == torch.arange(system.positions.values.shape[0])
        )

        neighbors = system.get_neighbors_list(self._nl_options)
        all_i = neighbors.samples.column("first_atom")
        all_j = neighbors.samples.column("second_atom")
        all_S = neighbors.samples.view(
            ["cell_shift_a", "cell_shift_b", "cell_shift_c"]
        ).values

        species = system.positions.samples.column("species")
        cell = system.cell.values.reshape(3, 3)
        positions = system.positions.values.reshape(-1, 3)

        per_atoms = run_options.outputs["energy"].per_atom
        if per_atoms:
            energy = torch.zeros(positions.shape[0], dtype=positions.dtype)
        else:
            energy = torch.zeros(1, dtype=positions.dtype)

        for i, j, S in zip(all_i, all_j, all_S):
            i = int(i)
            j = int(j)

            sigma, epsilon, shift = self._lj_params[int(species[i])][int(species[j])]
            distance = positions[j] - positions[i] + S.to(dtype=cell.dtype) @ cell
            r2 = distance.dot(distance)

            r6 = r2 * r2 * r2

            sigma3 = sigma * sigma * sigma
            sigma6 = sigma3 * sigma3

            sigma_r_6 = sigma6 / r6
            sigma_r_12 = sigma_r_6 * sigma_r_6

            e = 4.0 * epsilon * (sigma_r_12 - sigma_r_6) - shift

            if per_atoms:
                # We only compute each pair once (full_list=False in self._nl_options),
                # and assign half of the energy to each atom
                energy[i] += e / 2.0
                energy[j] += e / 2.0
            else:
                energy[0] += e

        if per_atoms:
            return {
                "energy": TensorBlock(
                    values=energy.reshape(-1, 1),
                    samples=system.positions.samples,
                    components=[],
                    properties=Labels(["energy"], torch.IntTensor([[0]])),
                )
            }
        else:
            return {
                "energy": TensorBlock(
                    values=energy.reshape(1, 1),
                    samples=Labels(["_"], torch.IntTensor([[0]])),
                    components=[],
                    properties=Labels(["energy"], torch.IntTensor([[0]])),
                )
            }

        return {}

    def requested_neighbors_lists(self) -> List[NeighborsListOptions]:
        return [self._nl_options]


CUTOFF = 5.0
SIGMA = 1.5808
EPSILON = 0.1729


@pytest.fixture
def model():
    model = LennardJones(cutoff=CUTOFF, params={(28, 28): (SIGMA, EPSILON)})
    model.train(False)

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        species=[28],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="ev",
                per_atom=False,
                forward_gradients=[],
            ),
        },
    )

    return MetatensorAtomisticModule(model, capabilities)


def check_against_ase_lj(atoms, calculator):
    ref = atoms.copy()
    ref.calc = ase.calculators.lj.LennardJones(
        sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF, ro=CUTOFF, smooth=False
    )

    atoms.calc = calculator

    assert np.allclose(ref.get_potential_energy(), atoms.get_potential_energy())
    assert np.allclose(ref.get_potential_energies(), atoms.get_potential_energies())
    assert np.allclose(ref.get_forces(), atoms.get_forces())
    assert np.allclose(ref.get_stress(), atoms.get_stress())


@pytest.fixture
def atoms():
    atoms = ase.build.make_supercell(
        ase.build.bulk("Ni", "fcc", a=3.6, cubic=True), 2 * np.eye(3)
    )
    atoms.positions += 0.2 * np.random.rand(*atoms.positions.shape)

    return atoms


def test_python_model(model, atoms):
    check_against_ase_lj(atoms, MetatensorCalculator(model))


def test_torch_script_model(model, atoms):
    model = torch.jit.script(model)
    check_against_ase_lj(atoms, MetatensorCalculator(model))


def test_exported_model(tmpdir, model, atoms):
    path = os.path.join(tmpdir, "exported-model.pt")
    model.export(path)
    check_against_ase_lj(atoms, MetatensorCalculator(path))
