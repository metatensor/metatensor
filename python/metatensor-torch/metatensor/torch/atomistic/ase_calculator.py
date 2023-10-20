import os
import pathlib
from typing import Dict, List, Union

import numpy as np
import torch

from .. import Labels, TensorBlock
from . import MetatensorAtomisticModule, ModelOutput, ModelRunOptions, System


import ase  # isort: skip
import ase.neighborlist  # isort: skip
from ase.calculators.calculator import (  # isort: skip
    Calculator,
    InputError,
    PropertyNotImplementedError,
    all_properties as ALL_ASE_PROPERTIES,
)


# import here to get an error early if the user is missing metatensor-operations
from .. import sum_over_samples_block  # isort: skip

FilePath = Union[str, bytes, pathlib.PurePath]


class MetatensorCalculator(Calculator):
    """
    The :py:class:`MetatensorCalculator` class implements ASE's ``Calculator`` API using
    metatensor atomistic models to compute energy, forces and any other supported
    property.

    This class can be initialized with any `MetatensorAtomisticModule`, and used to run
    simulations using ASE's MD facilities.
    """

    def __init__(
        self,
        model: Union[
            FilePath,
            torch.jit.RecursiveScriptModule,
            MetatensorAtomisticModule,
        ],
        check_consistency=False,
    ):
        """
        :param model: model to use for the calculation. This can be a file path, or a
            Python instance of :py:class:`MetatensorAtomisticModule`.
        :param check_consistency: should we check the model for consistency when
            running, defaults to False.
        """
        super().__init__()

        if isinstance(model, (str, bytes, pathlib.PurePath)):
            if not os.path.exists(model):
                raise InputError(f"given model path '{model}' does not exists")

            self._model = torch.jit.load(model)
        elif isinstance(model, torch.jit.RecursiveScriptModule):
            if model.original_name != "MetatensorAtomisticModule":
                raise InputError(
                    "torch model must be 'MetatensorAtomisticModule', "
                    f"got '{model.original_name}' instead"
                )
            self._model = model
        elif isinstance(model, MetatensorAtomisticModule):
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
        run_options: ModelRunOptions,
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
        positions, cell = _ase_to_torch_data(atoms)
        system = _torch_data_to_metatensor(atoms, positions, cell)

        # Compute the neighbors lists requested by the model using ASE NL
        for options in self._model.requested_neighbors_lists():
            system.add_neighbors_list(
                options, neighbors=_compute_ase_neighbors(atoms, options)
            )

        return self._model(
            system,
            run_options,
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
        positions, cell = _ase_to_torch_data(atoms)

        do_backward = False
        if "forces" in properties:
            do_backward = True
            positions.requires_grad_(True)

        if "stress" in properties:
            do_backward = True

            scaling = torch.eye(3, requires_grad=True, dtype=cell.dtype)

            positions = positions.reshape(-1, 3) @ scaling
            positions = positions.reshape(-1, 3, 1)
            positions.retain_grad()

            cell = cell.reshape(3, 3) @ scaling
            cell = cell.reshape(1, 3, 3, 1)

        if "stresses" in properties:
            raise NotImplementedError("'stresses' are not implemented yet")

        # convert from ase.Atoms to metatensor.torch.atomistic.System
        system = _torch_data_to_metatensor(atoms, positions, cell)
        for options in self._model.requested_neighbors_lists():
            system.add_neighbors_list(
                options, neighbors=_compute_ase_neighbors(atoms, options)
            )

        run_options = ModelRunOptions(
            length_unit="angstrom",
            selected_atoms=None,
            outputs=outputs,
        )

        outputs = self._model(
            system,
            run_options,
            check_consistency=self.parameters["check_consistency"],
        )
        energy = outputs["energy"]

        if run_options.outputs["energy"].per_atom:
            assert energy.values.shape == (len(atoms), 1)
            assert energy.samples == system.positions.samples
            energies = energy
            energy = sum_over_samples_block(energy, sample_names=["atom", "species"])
        else:
            assert energy.values.shape == (1, 1)

        assert len(energy.gradients_list()) == 0

        self.results = {}

        if "energies" in properties:
            self.results["energies"] = (
                energies.values.detach().to(device="cpu").numpy().reshape(-1)
            )

        assert energy.values.shape == (1, 1)
        if "energy" in properties:
            self.results["energy"] = (
                energy.values.detach().to(device="cpu").numpy()[0, 0]
            )

        if do_backward:
            energy.values.backward(-torch.ones_like(energy.values))

        if "forces" in properties:
            self.results["forces"] = (
                system.positions.values.grad.to(device="cpu").numpy().reshape(-1, 3)
            )

        if "stress" in properties:
            volume = atoms.cell.volume
            scaling_grad = -scaling.grad.to(device="cpu").numpy().reshape(3, 3)
            self.results["stress"] = scaling_grad / volume


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
    output.forward_gradients = []

    if "energies" in properties or "stresses" in properties:
        output.per_atom = True
    else:
        output.per_atom = False

    if "stresses" in properties:
        output.forward_gradients = ["cell"]

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

    positions = torch.from_numpy(atoms.positions).reshape(-1, 3, 1)

    if np.all(atoms.pbc):
        cell = torch.from_numpy(atoms.cell[:]).reshape(1, 3, 3, 1)
    elif np.any(atoms.pbc):
        raise ValueError(
            f"partial PBC ({atoms.pbc}) are not currently supported in "
            "metatensor atomistic models"
        )
    else:
        cell = torch.zeros((3, 3), dtype=torch.float64)

    return positions, cell


def _torch_data_to_metatensor(atoms, positions, cell):
    """
    Finish creating a ``System`` from ASE data. ``positions`` and ``cell`` should be
    created with ``_ase_to_torch_data()``.

    This is split into two different functions to allow modification of the
    positions/cell before constructing the full system.
    """
    positions = TensorBlock(
        values=positions,
        samples=Labels(
            ["atom", "species"],
            torch.IntTensor([(i, s) for i, s in enumerate(atoms.numbers)]),
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("position", 1),
    )

    cell = TensorBlock(
        values=cell.reshape(1, 3, 3, 1),
        samples=Labels.single(),
        components=[Labels.range("cell_abc", 3), Labels.range("xyz", 3)],
        properties=Labels.range("cell", 1),
    )

    return System(positions, cell)
