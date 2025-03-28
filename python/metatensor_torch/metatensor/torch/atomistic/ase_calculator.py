import logging
import os
import pathlib
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import vesin
from torch.profiler import record_function

from .. import Labels, TensorBlock, TensorMap
from . import (
    MetatensorAtomisticModel,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
    load_atomistic_model,
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


if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX", "0") == "0":
    # this can not be imported when building the documentation
    from .. import sum_over_samples  # isort: skip

FilePath = Union[str, bytes, pathlib.PurePath]

LOGGER = logging.getLogger(__name__)


STR_TO_DTYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
}


class MetatensorCalculator(ase.calculators.calculator.Calculator):
    """
    The :py:class:`MetatensorCalculator` class implements ASE's
    :py:class:`ase.calculators.calculator.Calculator` API using metatensor atomistic
    models to compute energy, forces and any other supported property.

    This class can be initialized with any :py:class:`MetatensorAtomisticModel`, and
    used to run simulations using ASE's MD facilities.

    Neighbor lists are computed using the fast
    `vesin <https://luthaf.fr/vesin/latest/index.html>`_ neighbor list library,
    unless the system has mixed periodic and non-periodic boundary conditions (which
    are not yet supported by ``vesin``), in which case the slower ASE neighbor list
    is used.
    """

    additional_outputs: Dict[str, TensorMap]
    """
    Additional outputs computed by :py:meth:`calculate` are stored in this dictionary.

    The keys will match the keys of the ``additional_outputs`` parameters to the
    constructor; and the values will be the corresponding raw
    :py:class:`metatensor.torch.TensorMap` produced by the model.
    """

    def __init__(
        self,
        model: Union[FilePath, MetatensorAtomisticModel],
        *,
        additional_outputs: Optional[Dict[str, ModelOutput]] = None,
        extensions_directory=None,
        check_consistency=False,
        device=None,
    ):
        """
        :param model: model to use for the calculation. This can be a file path, a
            Python instance of :py:class:`MetatensorAtomisticModel`, or the output of
            :py:func:`torch.jit.script` on :py:class:`MetatensorAtomisticModel`.
        :param additional_outputs: Dictionary of additional outputs to be computed by
            the model. These outputs will always be computed whenever the
            :py:meth:`calculate` function is called (e.g. by
            :py:meth:`ase.Atoms.get_potential_energy`,
            :py:meth:`ase.optimize.optimize.Dynamics.run`, *etc.*) and stored in the
            :py:attr:`additional_outputs` attribute. If you want more control over when
            and how to compute specific outputs, you should use :py:meth:`run_model`
            instead.
        :param extensions_directory: if the model uses extensions, we will try to load
            them from this directory
        :param check_consistency: should we check the model for consistency when
            running, defaults to False.
        :param device: torch device to use for the calculation. If ``None``, we will try
            the options in the model's ``supported_device`` in order.
        """
        super().__init__()

        self.parameters = {
            "check_consistency": check_consistency,
        }

        # Load the model
        if isinstance(model, (str, bytes, pathlib.PurePath)):
            if not os.path.exists(model):
                raise InputError(f"given model path '{model}' does not exist")

            self.parameters["model_path"] = str(model)

            model = load_atomistic_model(
                model, extensions_directory=extensions_directory
            )

        elif isinstance(model, torch.jit.RecursiveScriptModule):
            if model.original_name != "MetatensorAtomisticModel":
                raise InputError(
                    "torch model must be 'MetatensorAtomisticModel', "
                    f"got '{model.original_name}' instead"
                )
        elif isinstance(model, MetatensorAtomisticModel):
            # nothing to do
            pass
        else:
            raise TypeError(f"unknown type for model: {type(model)}")

        self.parameters["device"] = str(device) if device is not None else None
        # check if the model supports the requested device
        capabilities = model.capabilities()
        if device is None:
            device = _find_best_device(capabilities.supported_devices)
        else:
            device = torch.device(device)
            device_is_supported = False

            for supported in capabilities.supported_devices:
                try:
                    supported = torch.device(supported)
                except RuntimeError as e:
                    warnings.warn(
                        "the model contains an invalid device in `supported_devices`: "
                        f"{e}",
                        stacklevel=2,
                    )
                    continue

                if supported.type == device.type:
                    device_is_supported = True
                    break

            if not device_is_supported:
                raise ValueError(
                    f"This model does not support the requested device ({device}), "
                    "the following devices are supported: "
                    f"{capabilities.supported_devices}"
                )

        if capabilities.dtype in STR_TO_DTYPE:
            self._dtype = STR_TO_DTYPE[capabilities.dtype]
        else:
            raise ValueError(
                f"found unexpected dtype in model capabilities: {capabilities.dtype}"
            )

        if additional_outputs is None:
            self._additional_output_requests = {}
        else:
            assert isinstance(additional_outputs, dict)
            for name, output in additional_outputs.items():
                assert isinstance(name, str)
                assert isinstance(output, torch.ScriptObject)
                assert "explicit_gradients_setter" in output._method_names(), (
                    "outputs must be ModelOutput instances"
                )

            self._additional_output_requests = additional_outputs

        self._device = device
        self._model = model.to(device=self._device)

        # We do our own check to verify if a property is implemented in `calculate()`,
        # so we pretend to be able to compute all properties ASE knows about.
        self.implemented_properties = ALL_ASE_PROPERTIES

    def todict(self):
        if "model_path" not in self.parameters:
            raise RuntimeError(
                "can not save metatensor model in ASE `todict`, please initialize "
                "`MetatensorCalculator` with a path to a saved model file if you need "
                "to use `todict`"
            )

        return self.parameters

    @classmethod
    def fromdict(cls, data):
        return MetatensorCalculator(
            model=data["model_path"],
            check_consistency=data["check_consistency"],
            device=data["device"],
        )

    def metadata(self) -> ModelMetadata:
        """Get the metadata of the underlying model"""
        return self._model.metadata()

    def run_model(
        self,
        atoms: Union[ase.Atoms, List[ase.Atoms]],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """
        Run the model on the given ``atoms``, computing the requested ``outputs`` and
        only these.

        The output of the model is returned directly, and as such the blocks' ``values``
        will be :py:class:`torch.Tensor`.

        This is intended as an easy way to run metatensor models on
        :py:class:`ase.Atoms` when the model can compute outputs not supported by the
        standard ASE's calculator interface.

        All the parameters have the same meaning as the corresponding ones in
        :py:meth:`metatensor.torch.atomistic.ModelInterface.forward`.

        :param atoms: :py:class:`ase.Atoms`, or list of :py:class:`ase.Atoms`, on which
            to run the model
        :param outputs: outputs of the model that should be predicted
        :param selected_atoms: subset of atoms on which to run the calculation
        """
        if isinstance(atoms, ase.Atoms):
            atoms_list = [atoms]
        else:
            atoms_list = atoms

        systems = []
        for atoms in atoms_list:
            types, positions, cell, pbc = _ase_to_torch_data(
                atoms=atoms, dtype=self._dtype, device=self._device
            )
            system = System(types, positions, cell, pbc)
            # Compute the neighbors lists requested by the model
            for options in self._model.requested_neighbor_lists():
                neighbors = _compute_ase_neighbors(
                    atoms, options, dtype=self._dtype, device=self._device
                )
                register_autograd_neighbors(
                    system,
                    neighbors,
                    check_consistency=self.parameters["check_consistency"],
                )
                system.add_neighbor_list(options, neighbors)
            systems.append(system)

        options = ModelEvaluationOptions(
            length_unit="angstrom",
            outputs=outputs,
            selected_atoms=selected_atoms,
        )
        return self._model(
            systems=systems,
            options=options,
            check_consistency=self.parameters["check_consistency"],
        )

    def calculate(
        self,
        atoms: ase.Atoms,
        properties: List[str],
        system_changes: List[str],
    ) -> None:
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

        # In the next few lines, we decide which properties to calculate among energy,
        # forces and stress. In addition to the requested properties, we calculate the
        # energy if any of the three is requested, as it is an intermediate step in the
        # calculation of the other two. We also calculate the forces if the stress is
        # requested, and vice-versa. The overhead for the latter operation is also
        # small, assuming that the majority of the model computes forces and stresses
        # by backward propagation as opposed to forward-mode differentiation.
        calculate_energy = (
            "energy" in properties
            or "energies" in properties
            or "forces" in properties
            or "stress" in properties
        )
        calculate_energies = "energies" in properties
        calculate_forces = "forces" in properties or "stress" in properties
        calculate_stress = "stress" in properties
        if calculate_stress and not atoms.pbc.all():
            warnings.warn(
                "stress requested but likely to be wrong, since the system is not "
                "periodic in all directions",
                stacklevel=2,
            )
        if "forces" in properties and atoms.pbc.all():
            # we have PBCs, and, since the user/integrator requested forces, we will run
            # backward anyway, so let's do the stress as well for free (this saves
            # another forward-backward call later if the stress is requested)
            calculate_stress = True
        if "stresses" in properties:
            raise NotImplementedError("'stresses' are not implemented yet")

        with record_function("ASECalculator::prepare_inputs"):
            outputs = _ase_properties_to_metatensor_outputs(properties)
            outputs.update(self._additional_output_requests)

            capabilities = self._model.capabilities()
            for name in outputs.keys():
                if name not in capabilities.outputs:
                    raise ValueError(
                        f"you asked for the calculation of {name}, but this model "
                        "does not support it"
                    )

            types, positions, cell, pbc = _ase_to_torch_data(
                atoms=atoms, dtype=self._dtype, device=self._device
            )

            do_backward = False
            if calculate_forces:
                do_backward = True
                positions.requires_grad_(True)

            if calculate_stress:
                do_backward = True

                strain = torch.eye(
                    3, requires_grad=True, device=self._device, dtype=self._dtype
                )

                positions = positions @ strain
                positions.retain_grad()

                cell = cell @ strain

            run_options = ModelEvaluationOptions(
                length_unit="angstrom",
                outputs=outputs,
                selected_atoms=None,
            )

        with record_function("ASECalculator::compute_neighbors"):
            # convert from ase.Atoms to metatensor.torch.atomistic.System
            system = System(types, positions, cell, pbc)

            for options in self._model.requested_neighbor_lists():
                neighbors = _compute_ase_neighbors(
                    atoms, options, dtype=self._dtype, device=self._device
                )
                register_autograd_neighbors(
                    system,
                    neighbors,
                    check_consistency=self.parameters["check_consistency"],
                )
                system.add_neighbor_list(options, neighbors)

        # no `record_function` here, this will be handled by MetatensorAtomisticModel
        outputs = self._model(
            [system],
            run_options,
            check_consistency=self.parameters["check_consistency"],
        )
        energy = outputs["energy"]

        with record_function("ASECalculator::sum_energies"):
            if run_options.outputs["energy"].per_atom:
                assert len(energy) == 1
                assert energy.sample_names == ["system", "atom"]
                assert torch.all(energy.block().samples["system"] == 0)
                assert torch.all(
                    energy.block().samples["atom"] == torch.arange(positions.shape[0])
                )
                energies = energy.block().values
                assert energies.shape == (len(atoms), 1)

                energy = sum_over_samples(energy, sample_names=["atom"])

            assert len(energy.block().gradients_list()) == 0
            energy = energy.block().values
            assert energy.shape == (1, 1)

        with record_function("ASECalculator::run_backward"):
            if do_backward:
                energy.backward()

        with record_function("ASECalculator::convert_outputs"):
            self.results = {}

            if calculate_energies:
                energies_values = energies.detach().reshape(-1)
                energies_values = energies_values.to(device="cpu").to(
                    dtype=torch.float64
                )
                self.results["energies"] = energies_values.numpy()

            if calculate_energy:
                energy_values = energy.detach()
                energy_values = energy_values.to(device="cpu").to(dtype=torch.float64)
                self.results["energy"] = energy_values.numpy()[0, 0]

            if calculate_forces:
                forces_values = -system.positions.grad.reshape(-1, 3)
                forces_values = forces_values.to(device="cpu").to(dtype=torch.float64)
                self.results["forces"] = forces_values.numpy()

            if calculate_stress:
                stress_values = strain.grad.reshape(3, 3) / atoms.cell.volume
                stress_values = stress_values.to(device="cpu").to(dtype=torch.float64)
                self.results["stress"] = _full_3x3_to_voigt_6_stress(
                    stress_values.numpy()
                )

            self.additional_outputs = {}
            for name in self._additional_output_requests:
                self.additional_outputs[name] = outputs[name]

    def compute_energy(
        self,
        atoms: Union[ase.Atoms, List[ase.Atoms]],
        compute_forces_and_stresses: bool = False,
    ) -> Dict[str, Union[Union[float, np.ndarray], List[Union[float, np.ndarray]]]]:
        """
        Compute the energy of the given ``atoms``.

        Energies are computed in eV, forces in eV/Å, and stresses in 3x3 tensor format
        and in units of eV/Å^3.

        :param atoms: :py:class:`ase.Atoms`, or list of :py:class:`ase.Atoms`, on which
            to run the model
        :param compute_forces_and_stresses: if ``True``, the model will also compute
            forces and stresses. IMPORTANT: stresses will only be computed if all
            provided systems have periodic boundary conditions in all directions.

        :return: A dictionary with the computed properties. The dictionary will contain
            the ``energy`` as a float, and, if requested, the ``forces`` and ``stress``
            as numpy arrays. In case of a list of :py:class:`ase.Atoms`, the dictionary
            values will instead be lists of the corresponding properties, in the same
            format.
        """
        if "energy" not in self._model.capabilities().outputs:
            raise ValueError("this model does not support energy computation")

        outputs = {"energy": ModelOutput(quantity="energy", unit="eV")}

        if isinstance(atoms, ase.Atoms):
            atoms_list = [atoms]
            was_single = True
        else:
            atoms_list = atoms
            was_single = False

        systems = []
        if compute_forces_and_stresses:
            strains = []
        for atoms in atoms_list:
            types, positions, cell, pbc = _ase_to_torch_data(
                atoms=atoms, dtype=self._dtype, device=self._device
            )
            if compute_forces_and_stresses:
                positions.requires_grad_(True)
                strain = torch.eye(
                    3, requires_grad=True, device=self._device, dtype=self._dtype
                )
                positions = positions @ strain
                positions.retain_grad()
                cell = cell @ strain
                strains.append(strain)
            system = System(types, positions, cell, pbc)
            # Compute the neighbors lists requested by the model
            for options in self._model.requested_neighbor_lists():
                neighbors = _compute_ase_neighbors(
                    atoms, options, dtype=self._dtype, device=self._device
                )
                register_autograd_neighbors(
                    system,
                    neighbors,
                    check_consistency=self.parameters["check_consistency"],
                )
                system.add_neighbor_list(options, neighbors)
            systems.append(system)

        options = ModelEvaluationOptions(
            length_unit="angstrom",
            outputs=outputs,
        )
        energies = self._model(
            systems=systems,
            options=options,
            check_consistency=self.parameters["check_consistency"],
        )["energy"]

        results_as_numpy_arrays = {
            "energy": energies.block().values.detach().cpu().numpy().flatten().tolist()
        }
        if compute_forces_and_stresses:
            energy_tensor = energies.block().values
            energy_tensor.backward(torch.ones_like(energy_tensor))
            results_as_numpy_arrays["forces"] = [
                -system.positions.grad.cpu().numpy() for system in systems
            ]
            if all(atoms.pbc.all() for atoms in atoms_list):
                results_as_numpy_arrays["stress"] = [
                    strain.grad.cpu().numpy() / atoms.cell.volume
                    for strain, atoms in zip(strains, atoms_list)
                ]
        if was_single:
            for key, value in results_as_numpy_arrays.items():
                results_as_numpy_arrays[key] = value[0]
        return results_as_numpy_arrays


def _find_best_device(devices: List[str]) -> torch.device:
    """
    Find the best device from the list of ``devices`` that is available to the current
    PyTorch installation.
    """

    for device in devices:
        if device == "cpu":
            return torch.device("cpu")
        elif device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                LOGGER.warning(
                    "the model suggested to use CUDA devices before CPU, "
                    "but we are unable to find it"
                )
        elif device == "mps":
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_built()
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            else:
                LOGGER.warning(
                    "the model suggested to use MPS devices before CPU, "
                    "but we are unable to find it"
                )
        else:
            warnings.warn(
                f"unknown device in the model's `supported_devices`: '{device}'",
                stacklevel=2,
            )

    warnings.warn(
        "could not find a valid device in the model's `supported_devices`, "
        "falling back to CPU",
        stacklevel=2,
    )
    return torch.device("cpu")


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


def _compute_ase_neighbors(atoms, options, dtype, device):
    # options.strict is ignored by this function, since `ase.neighborlist.neighbor_list`
    # only computes strict NL, and these are valid even with `strict=False`

    if np.all(atoms.pbc) or np.all(~atoms.pbc):
        nl_i, nl_j, nl_S, nl_D = vesin.ase_neighbor_list(
            "ijSD",
            atoms,
            cutoff=options.engine_cutoff(engine_length_unit="angstrom"),
        )
    else:
        nl_i, nl_j, nl_S, nl_D = ase.neighborlist.neighbor_list(
            "ijSD",
            atoms,
            cutoff=options.engine_cutoff(engine_length_unit="angstrom"),
        )

    # The pair selection code here below avoids a relatively slow loop over
    # all pairs to improve performance
    reject_condition = (
        # we want a half neighbor list, so drop all duplicated neighbors
        (nl_j < nl_i)
        | (
            (nl_i == nl_j)
            & (
                # only create pairs with the same atom twice if the pair spans more
                # than one unit cell
                ((nl_S[:, 0] == 0) & (nl_S[:, 1] == 0) & (nl_S[:, 2] == 0))
                |
                # When creating pairs between an atom and one of its periodic images,
                # the code generates multiple redundant pairs
                # (e.g. with shifts 0 1 1 and 0 -1 -1); and we want to only keep one of
                # these. We keep the pair in the positive half plane of shifts.
                (
                    (nl_S.sum(axis=1) < 0)
                    | (
                        (nl_S.sum(axis=1) == 0)
                        & ((nl_S[:, 2] < 0) | ((nl_S[:, 2] == 0) & (nl_S[:, 1] < 0)))
                    )
                )
            )
        )
    )
    selected = np.logical_not(reject_condition)
    n_pairs = np.sum(selected)

    if options.full_list:
        distances = np.empty((2 * n_pairs, 3), dtype=np.float64)
        samples = np.empty((2 * n_pairs, 5), dtype=np.int32)
    else:
        distances = np.empty((n_pairs, 3), dtype=np.float64)
        samples = np.empty((n_pairs, 5), dtype=np.int32)

    samples[:n_pairs, 0] = nl_i[selected]
    samples[:n_pairs, 1] = nl_j[selected]
    samples[:n_pairs, 2:] = nl_S[selected]

    distances[:n_pairs] = nl_D[selected]

    if options.full_list:
        samples[n_pairs:, 0] = nl_j[selected]
        samples[n_pairs:, 1] = nl_i[selected]
        samples[n_pairs:, 2:] = -nl_S[selected]

        distances[n_pairs:] = -nl_D[selected]

    distances = torch.from_numpy(distances).to(dtype=dtype).to(device=device)
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
            values=torch.from_numpy(samples),
        ).to(device=device),
        components=[Labels.range("xyz", 3).to(device)],
        properties=Labels.range("distance", 1).to(device),
    )


def _ase_to_torch_data(atoms, dtype, device):
    """Get the positions, cell and pbc from ASE atoms as torch tensors"""

    types = torch.from_numpy(atoms.numbers).to(dtype=torch.int32, device=device)
    positions = torch.from_numpy(atoms.positions).to(dtype=dtype, device=device)
    cell = torch.zeros((3, 3), dtype=dtype, device=device)
    pbc = torch.tensor(atoms.pbc, dtype=torch.bool, device=device)

    cell[pbc] = torch.tensor(atoms.cell[atoms.pbc], dtype=dtype, device=device)

    return types, positions, cell, pbc


def _full_3x3_to_voigt_6_stress(stress):
    """
    Re-implementation of ``ase.stress.full_3x3_to_voigt_6_stress`` which does not do the
    stress symmetrization correctly (they do ``(stress[1, 2] + stress[1, 2]) / 2.0``)
    """
    return np.array(
        [
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            (stress[1, 2] + stress[2, 1]) / 2.0,
            (stress[0, 2] + stress[2, 0]) / 2.0,
            (stress[0, 1] + stress[1, 0]) / 2.0,
        ]
    )
