import os
from typing import Dict, List, Optional

import numpy as np
import pytest
import torch

from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelOutput,
    NeighborsListOptions,
    System,
)
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator


ase = pytest.importorskip("ase")
import ase.build  # noqa  isort: skip
import ase.units  # noqa  isort: skip
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
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        if "energy" not in outputs:
            return {}

        if selected_atoms is not None:
            # TODO: add selected_atoms
            raise NotImplementedError("selected atoms is not yet implemented")

        per_atoms = outputs["energy"].per_atom

        all_energies = []
        for system in systems:
            neighbors = system.get_neighbors_list(self._nl_options)
            all_i = neighbors.samples.column("first_atom")
            all_j = neighbors.samples.column("second_atom")

            species = system.species
            positions = system.positions

            if per_atoms:
                energy = torch.zeros(positions.shape[0], dtype=positions.dtype)
            else:
                energy = torch.zeros(1, dtype=positions.dtype)

            for i, j, distance in zip(all_i, all_j, neighbors.values.reshape(-1, 3)):
                sigma, epsilon, shift = self._lj_params[int(species[i])][
                    int(species[j])
                ]
                r2 = distance.dot(distance)

                r6 = r2 * r2 * r2

                sigma3 = sigma * sigma * sigma
                sigma6 = sigma3 * sigma3

                sigma_r_6 = sigma6 / r6
                sigma_r_12 = sigma_r_6 * sigma_r_6

                e = 4.0 * epsilon * (sigma_r_12 - sigma_r_6) - shift

                if per_atoms:
                    # We only compute each pair once (full_list=False in
                    # self._nl_options), and assign half of the energy to each atom
                    energy[i] += e / 2.0
                    energy[j] += e / 2.0
                else:
                    energy[0] += e

            all_energies.append(energy)

        if per_atoms:
            samples_list: List[List[int]] = []
            for s, system in enumerate(systems):
                for a in range(len(system)):
                    samples_list.append([s, a])

            samples = Labels(
                ["structure", "atom"],
                torch.tensor(samples_list, dtype=torch.int32),
            )
        else:
            samples = Labels(
                ["structure"],
                torch.tensor([[s] for s in range(len(systems))], dtype=torch.int32),
            )

        block = TensorBlock(
            values=torch.vstack(all_energies).reshape(-1, 1),
            samples=samples,
            components=[],
            properties=Labels(["energy"], torch.IntTensor([[0]])),
        )
        return {
            "energy": TensorMap(Labels(["_"], torch.IntTensor([[0]])), [block]),
        }

    def requested_neighbors_lists(self) -> List[NeighborsListOptions]:
        return [self._nl_options]


CUTOFF = 5.0
SIGMA = 1.5808
EPSILON = 0.1729


@pytest.fixture
def model():
    model = LennardJones(
        cutoff=CUTOFF,
        params={(28, 28): (SIGMA, EPSILON)},
    )
    model.train(False)

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=[28],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
                per_atom=False,
                explicit_gradients=[],
            ),
        },
    )

    return MetatensorAtomisticModel(model, capabilities)


@pytest.fixture
def model_different_units():
    model = LennardJones(
        cutoff=CUTOFF * ase.units.Bohr,
        params={
            (28, 28): (
                SIGMA * ase.units.Bohr,
                EPSILON * ase.units.kJ / ase.units.mol,
            )
        },
    )
    model.train(False)

    capabilities = ModelCapabilities(
        length_unit="Bohr",
        species=[28],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="kJ/mol",
                per_atom=False,
                explicit_gradients=[],
            ),
        },
    )

    return MetatensorAtomisticModel(model, capabilities)


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
    np.random.seed(0xDEADBEEF)

    atoms = ase.build.make_supercell(
        ase.build.bulk("Ni", "fcc", a=3.6, cubic=True), 2 * np.eye(3)
    )
    atoms.positions += 0.2 * np.random.rand(*atoms.positions.shape)

    return atoms


def test_python_model(model, model_different_units, atoms):
    check_against_ase_lj(atoms, MetatensorCalculator(model))
    check_against_ase_lj(atoms, MetatensorCalculator(model_different_units))


def test_torch_script_model(model, model_different_units, atoms):
    model = torch.jit.script(model)
    check_against_ase_lj(atoms, MetatensorCalculator(model))

    model_different_units = torch.jit.script(model_different_units)
    check_against_ase_lj(atoms, MetatensorCalculator(model_different_units))


def test_exported_model(tmpdir, model, model_different_units, atoms):
    path = os.path.join(tmpdir, "exported-model.pt")
    model.export(path)
    check_against_ase_lj(atoms, MetatensorCalculator(path))

    model_different_units.export(path)
    check_against_ase_lj(atoms, MetatensorCalculator(path))
