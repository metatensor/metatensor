import os
import pathlib
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .. import Labels, TensorBlock
from . import (
    MetatensorAtomisticModel,
    ModelEvaluationOptions,
    ModelOutput,
    System,
    check_atomistic_model,
    register_autograd_neighbors,
)


import ase  # isort: skip
import ase.neighborlist  # isort: skip
import ase.calculators.calculator  # isort: skip
from ase.calculators.calculator import (  # isort: skip
    InputError,
    PropertyNotImplementedError,
    all_properties as ALL_ASE_PROPERTIES,
)


# import here to get an error early if the user is missing metatensor-operations
from .. import sum_over_samples  # isort: skip

FilePath = Union[str, bytes, pathlib.PurePath]


class MetatensorCalculator(ase.calculators.calculator.Calculator):
    """
    The :py:class:`MetatensorCalculator` class implements ASE's
    :py:class:`ase.calculators.calculator.Calculator` API using metatensor atomistic
    models to compute energy, forces and any other supported property.

    This class can be initialized with any :py:class:`MetatensorAtomisticModel`, and
    used to run simulations using ASE's MD facilities.
    """

    def __init__(
        self,
        model: Union[
            FilePath,
            MetatensorAtomisticModel,
        ],
        check_consistency=False,
    ):
        """
        :param model: model to use for the calculation. This can be a file path, a
            Python instance of :py:class:`MetatensorAtomisticModel`, or the output of
            :py:func:`torch.jit.script` on :py:class:`MetatensorAtomisticModel`.
        :param check_consistency: should we check the model for consistency when
            running, defaults to False.
        """
        super().__init__()

        if isinstance(model, (str, bytes, pathlib.PurePath)):
            if not os.path.exists(model):
                raise InputError(f"given model path '{model}' does not exists")

            check_atomistic_model(model)
            self._model = torch.jit.load(model)
        elif isinstance(model, torch.jit.RecursiveScriptModule):
            if model.original_name != "MetatensorAtomisticModel":
                raise InputError(
                    "torch model must be 'MetatensorAtomisticModel', "
                    f"got '{model.original_name}' instead"
                )
            self._model = model
        elif isinstance(model, MetatensorAtomisticModel):
            self._model = model
        else:
            raise TypeError(f"unknown type for model: {type(model)}")

        self.parameters = {
            "model": model,
            "check_consistency": check_consistency,
        }

        # We do our own check to verify if a property is implemented in `calculate()`,
        # so we pretend to be able to compute all properties ASE knows about.
        self.implemented_properties = ALL_ASE_PROPERTIES

    def todict(self):
        # used by ASE to save the calculator
        raise NotImplementedError("todict is not yet implemented")

    @classmethod
    def fromdict(cls, dict):
        # used by ASE to load a saved calculator
        raise NotImplementedError("fromdict is not yet implemented")

    def run_model(
        self,
        atoms: ase.Atoms,
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorBlock]:
        """
        Run the model on the given ``atoms``, computing properties according to the
        ``outputs`` and ``selected_atoms`` options.

        The output of the model is returned directly, and as such the blocks' ``values``
        will be :py:class:`torch.Tensor`.

        This is intended as an easy way to run metatensor models on
        :py:class:`ase.Atoms` when the model can predict properties not supported by the
        usual ASE's calculator interface.
        """
        species, positions, cell = _ase_to_torch_data(atoms)
        system = System(species, positions, cell)

        # Compute the neighbors lists requested by the model using ASE NL
        for options in self._model.requested_neighbors_lists(length_unit="angstrom"):
            neighbors = _compute_ase_neighbors(atoms, options)
            register_autograd_neighbors(
                system,
                neighbors,
                check_consistency=self.parameters["check_consistency"],
            )
            system.add_neighbors_list(options, neighbors)

        options = ModelEvaluationOptions(
            length_unit="angstrom",
            outputs=outputs,
            selected_atoms=selected_atoms,
        )
        return self._model(
            systems=[system],
            options=options,
            check_consistency=self.parameters["check_consistency"],
        )

    def calculate(
        self,
        atoms: ase.Atoms,
        properties: List[str],
        system_changes: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Compute some ``properties`` with this calculator, and return them in the format
        expected by ASE.

        This is not intended to be called directly by users, but to be an implementation
        detail of ``atoms.get_energy()`` and related functions. See
        :py:meth:`ase.calculators.calculator.Calculator.calculate` for more information.
        """
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )

        outputs = _ase_properties_to_metatensor_outputs(properties)
        species, positions, cell = _ase_to_torch_data(atoms)

        do_backward = False
        if "forces" in properties:
            do_backward = True
            positions.requires_grad_(True)

        if "stress" in properties:
            do_backward = True

            scaling = torch.eye(3, requires_grad=True, dtype=cell.dtype)

            positions = positions @ scaling
            positions.retain_grad()

            cell = cell @ scaling

        if "stresses" in properties:
            raise NotImplementedError("'stresses' are not implemented yet")

        # convert from ase.Atoms to metatensor.torch.atomistic.System
        system = System(species, positions, cell)
        for options in self._model.requested_neighbors_lists(length_unit="angstrom"):
            neighbors = _compute_ase_neighbors(atoms, options)
            register_autograd_neighbors(
                system,
                neighbors,
                check_consistency=self.parameters["check_consistency"],
            )
            system.add_neighbors_list(options, neighbors)

        run_options = ModelEvaluationOptions(
            length_unit="angstrom",
            outputs=outputs,
            selected_atoms=None,
        )

        outputs = self._model(
            [system],
            run_options,
            check_consistency=self.parameters["check_consistency"],
        )
        energy = outputs["energy"]

        if run_options.outputs["energy"].per_atom:
            assert len(energy) == 1
            assert energy.sample_names == ["structure", "atom"]
            assert torch.all(energy.block().samples["structure"] == 0)
            assert torch.all(
                energy.block().samples["atom"] == torch.arange(positions.shape[0])
            )
            energies = energy.block().values
            assert energies.shape == (len(atoms), 1)

            energy = sum_over_samples(energy, sample_names=["atom"])

        assert len(energy.block().gradients_list()) == 0
        energy = energy.block().values
        assert energy.shape == (1, 1)

        self.results = {}

        if "energies" in properties:
            self.results["energies"] = (
                energies.detach().to(device="cpu").numpy().reshape(-1)
            )

        if "energy" in properties:
            self.results["energy"] = energy.detach().to(device="cpu").numpy()[0, 0]

        if do_backward:
            energy.backward(-torch.ones_like(energy))

        if "forces" in properties:
            self.results["forces"] = (
                system.positions.grad.to(device="cpu").numpy().reshape(-1, 3)
            )

        if "stress" in properties:
            volume = atoms.cell.volume
            scaling_grad = -scaling.grad.to(device="cpu").numpy().reshape(3, 3)
            self.results["stress"] = _full_3x3_to_voigt_6_stress(scaling_grad / volume)


def _ase_properties_to_metatensor_outputs(properties):
    energy_properties = []
    for p in properties:
        if p in ["energy", "energies", "forces", "stress", "stresses"]:
            energy_properties.append(p)
        else:
            raise PropertyNotImplementedError(
                f"property '{p}' it not yet supported by this calculator, "
                "even if it might be supported by the model"
            )

    output = ModelOutput()
    output.quantity = "energy"
    output.unit = "ev"
    output.explicit_gradients = []

    if "energies" in properties or "stresses" in properties:
        output.per_atom = True
    else:
        output.per_atom = False

    if "stresses" in properties:
        output.explicit_gradients = ["cell"]

    return {"energy": output}


def _compute_ase_neighbors(atoms, options):
    nl = ase.neighborlist.NeighborList(
        cutoffs=[options.engine_cutoff] * len(atoms),
        skin=0.0,
        sorted=False,
        self_interaction=False,
        bothways=options.full_list,
        primitive=ase.neighborlist.NewPrimitiveNeighborList,
    )
    nl.update(atoms)

    cell = torch.from_numpy(atoms.cell[:])
    positions = torch.from_numpy(atoms.positions)

    samples = []
    distances = []
    cutoff2 = options.engine_cutoff * options.engine_cutoff
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            distance = positions[j] - positions[i] + offset.dot(cell)

            distance2 = torch.dot(distance, distance).item()

            if distance2 > cutoff2:
                continue

            samples.append((i, j, offset[0], offset[1], offset[2]))
            distances.append(distance.to(dtype=torch.float64))

    if len(distances) == 0:
        distances = torch.zeros((0, 3), dtype=positions.dtype)
        samples = torch.zeros((0, 5), dtype=torch.int32)
    else:
        samples = torch.tensor(samples, dtype=torch.int32)
        distances = torch.vstack(distances)

    return TensorBlock(
        values=distances.reshape(-1, 3, 1),
        samples=Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=samples,
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )


def _ase_to_torch_data(atoms):
    """Get the positions and cell from ASE atoms as torch tensors"""

    species = torch.from_numpy(atoms.numbers).to(dtype=torch.int32)
    positions = torch.from_numpy(atoms.positions)

    if np.all(atoms.pbc):
        cell = torch.from_numpy(atoms.cell[:])
    elif np.any(atoms.pbc):
        raise ValueError(
            f"partial PBC ({atoms.pbc}) are not currently supported in "
            "metatensor atomistic models"
        )
    else:
        cell = torch.zeros((3, 3), dtype=torch.float64)

    return species, positions, cell


def _full_3x3_to_voigt_6_stress(stress):
    ase.stress.full_3x3_to_voigt_6_stress
    """
    Re-implementation of ``ase.stress.full_3x3_to_voigt_6_stress`` which does not do the
    stress symmetrization correctly (they do ``(stress[1, 2] + stress[1, 2]) / 2.0``)
    """
    return np.transpose(
        [
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            (stress[1, 2] + stress[2, 1]) / 2.0,
            (stress[0, 2] + stress[2, 0]) / 2.0,
            (stress[0, 1] + stress[1, 0]) / 2.0,
        ]
    )
