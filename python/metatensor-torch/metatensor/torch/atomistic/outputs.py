from typing import Dict, List, Optional

import torch

from .. import Labels, TensorMap, dtype_name
from . import ModelOutput, System


def _check_outputs(
    systems: List[System],
    requested: Dict[str, ModelOutput],
    selected_atoms: Optional[Labels],
    outputs: Dict[str, TensorMap],
    expected_dtype: torch.dtype,
):
    """
    Check that the outputs of a model conform to the expected structure for metatensor
    atomistic models.

    This function checks conformance with the reference documentation in
    https://docs.metatensor.org/latest/atomistic/outputs.html
    """

    for name, output in outputs.items():
        if requested.get(name) is None:
            raise ValueError(
                f"the model produced an output named '{name}', which was not requested"
            )

        if len(output) != 0:
            output_dtype = output.block_by_id(0).values.dtype
            if output_dtype != expected_dtype:
                raise ValueError(
                    f"wrong dtype for the {name} output: "
                    f"the model promised {dtype_name(expected_dtype)}, "
                    f"we got {dtype_name(output_dtype)}"
                )

    for name, request in requested.items():
        value = outputs.get(name)
        if value is None:
            raise ValueError(
                f"the model did not produce the '{name}' output, which was requested"
            )

        if name == "energy":
            _check_energy(systems, request, selected_atoms, energy=value)
        elif name == "energy_ensemble":
            _check_energy_ensemble(
                systems, request, selected_atoms, energy_ensemble=value
            )
        else:
            # this is a non-standard output, there is nothing to check
            continue


def _check_energy(
    systems: List[System],
    request: ModelOutput,
    selected_atoms: Optional[Labels],
    energy: TensorMap,
):
    """Check the "energy" output metadata"""
    if energy.keys != Labels("_", torch.tensor([[0]])):
        raise ValueError(
            "invalid keys for 'energy' output: expected `Labels('_', [[0]])`"
        )

    device = energy.device
    energy_block = energy.block_by_id(0)

    if request.per_atom:
        expected_samples_names = ["system", "atom"]
    else:
        expected_samples_names = ["system"]

    if energy_block.samples.names != expected_samples_names:
        raise ValueError(
            "invalid sample names for 'energy' output: "
            f"expected {expected_samples_names}, got {energy_block.samples.names}"
        )

    # check samples values from systems & selected_atoms
    if request.per_atom:
        expected_values: List[List[int]] = []
        for s, system in enumerate(systems):
            for a in range(len(system)):
                expected_values.append([s, a])

        expected_samples = Labels(
            ["system", "atom"], torch.tensor(expected_values, device=device)
        )
        if selected_atoms is not None:
            expected_samples = expected_samples.intersection(selected_atoms)

        if len(expected_samples.union(energy_block.samples)) != len(expected_samples):
            raise ValueError(
                "invalid samples entries for 'energy' output, they do not match the "
                f"`systems` and `selected_atoms`. Expected samples:\n{expected_samples}"
            )

    else:
        expected_samples = Labels(
            "system", torch.arange(len(systems), device=device).reshape(-1, 1)
        )
        if selected_atoms is not None:
            selected_systems = Labels(
                "system", torch.unique(selected_atoms.column("system")).reshape(-1, 1)
            )
            expected_samples = expected_samples.intersection(selected_systems)

        if len(expected_samples.union(energy_block.samples)) != len(expected_samples):
            raise ValueError(
                "invalid samples entries for 'energy' output, they do not match the "
                f"`systems` and `selected_atoms`. Expected samples:\n{expected_samples}"
            )

    if len(energy_block.components) != 0:
        raise ValueError(
            "invalid components for 'energy' output: components should be empty"
        )

    if energy_block.properties != Labels("energy", torch.tensor([[0]], device=device)):
        raise ValueError(
            "invalid properties for 'energy' output: expected `Labels('energy', [[0]])`"
        )

    for parameter, gradient in energy_block.gradients():
        if parameter not in ["strain", "positions"]:
            raise ValueError(f"invalid gradient for 'energy' output: {parameter}")

        xyz = torch.tensor([[0], [1], [2]], device=device)
        # strain gradient checks
        if parameter == "strain":
            if gradient.samples.names != ["sample"]:
                raise ValueError(
                    "invalid samples for 'energy' output 'strain' gradients: "
                    f"expected the names to be ['sample'], got {gradient.samples.names}"
                )

            if len(gradient.components) != 2:
                raise ValueError(
                    "invalid components for 'energy' output 'strain' gradients: "
                    "expected two components"
                )

            if gradient.components[0] != Labels("xyz_1", xyz):
                raise ValueError(
                    "invalid components for 'energy' output 'strain' gradients: "
                    "expected Labels('xyz_1', [[0], [1], [2]]) for the first component"
                )

            if gradient.components[1] != Labels("xyz_2", xyz):
                raise ValueError(
                    "invalid components for 'energy' output 'strain' gradients: "
                    "expected Labels('xyz_2', [[0], [1], [2]]) for the second component"
                )

        # positions gradient checks
        if parameter == "positions":
            if gradient.samples.names != ["sample", "system", "atom"]:
                raise ValueError(
                    "invalid samples for 'energy' output 'positions' gradients: "
                    "expected the names to be ['sample', 'system', 'atom'], "
                    f"got {gradient.samples.names}"
                )

            if len(gradient.components) != 1:
                raise ValueError(
                    "invalid components for 'energy' output 'positions' gradients: "
                    "expected one component"
                )

            if gradient.components[0] != Labels("xyz", xyz):
                raise ValueError(
                    "invalid components for 'energy' output 'positions' gradients: "
                    "expected Labels('xyz', [[0], [1], [2]]) for the first component"
                )


def _check_energy_ensemble(
    systems: List[System],
    request: ModelOutput,
    selected_atoms: Optional[Labels],
    energy_ensemble: TensorMap,
):
    """Check the "energy_ensemble" output metadata"""
    if energy_ensemble.keys != Labels("_", torch.tensor([[0]])):
        raise ValueError(
            "invalid keys for 'energy_ensemble' output: expected `Labels('_', [[0]])`"
        )

    device = energy_ensemble.device
    energy_ensemble_block = energy_ensemble.block_by_id(0)

    if request.per_atom:
        expected_samples_names = ["system", "atom"]
    else:
        expected_samples_names = ["system"]

    if energy_ensemble_block.samples.names != expected_samples_names:
        raise ValueError(
            "invalid sample names for 'energy_ensemble' output: "
            f"expected {expected_samples_names}, got "
            f"{energy_ensemble_block.samples.names}"
        )

    # check samples values from systems & selected_atoms
    if request.per_atom:
        expected_values: List[List[int]] = []
        for s, system in enumerate(systems):
            for a in range(len(system)):
                expected_values.append([s, a])

        expected_samples = Labels(
            ["system", "atom"], torch.tensor(expected_values, device=device)
        )
        if selected_atoms is not None:
            expected_samples = expected_samples.intersection(selected_atoms)

        if len(expected_samples.union(energy_ensemble_block.samples)) != len(
            expected_samples
        ):
            raise ValueError(
                "invalid samples entries for 'energy_ensemble_block' output, they "
                "do not match the `systems` and `selected_atoms`. "
                f"Expected samples:\n{expected_samples}"
            )

    else:
        expected_samples = Labels(
            "system", torch.arange(len(systems), device=device).reshape(-1, 1)
        )
        if selected_atoms is not None:
            selected_systems = Labels(
                "system", torch.unique(selected_atoms.column("system")).reshape(-1, 1)
            )
            expected_samples = expected_samples.intersection(selected_systems)

        if len(expected_samples.union(energy_ensemble_block.samples)) != len(
            expected_samples
        ):
            raise ValueError(
                "invalid samples entries for 'energy_ensemble' output, they do not "
                "match the `systems` and `selected_atoms`. "
                f"Expected samples:\n{expected_samples}"
            )

    if len(energy_ensemble_block.components) != 0:
        raise ValueError(
            "invalid components for 'energy_ensemble' output: components "
            "should be empty"
        )

    if energy_ensemble_block.properties != Labels(
        "ensemble_member",
        torch.arange(energy_ensemble_block.values.shape[1], device=device).reshape(
            -1, 1
        ),
    ):
        raise ValueError(
            "invalid properties for 'energy_ensemble' output: expected a Labels "
            "object with the name 'ensemble_member' and the values being the indices "
            "of the ensemble members from 0 to the number of ensemble members minus "
            "one."
        )

    for parameter, gradient in energy_ensemble_block.gradients():
        if parameter not in ["strain", "positions"]:
            raise ValueError(
                f"invalid gradient for 'energy_ensemble' output: {parameter}"
            )

        xyz = torch.tensor([[0], [1], [2]], device=device)
        # strain gradient checks
        if parameter == "strain":
            if gradient.samples.names != ["sample"]:
                raise ValueError(
                    "invalid samples for 'energy_ensemble' output 'strain' gradients: "
                    f"expected the names to be ['sample'], got {gradient.samples.names}"
                )

            if len(gradient.components) != 2:
                raise ValueError(
                    "invalid components for 'energy_ensemble' output 'strain' "
                    "gradients: expected two components"
                )

            if gradient.components[0] != Labels("xyz_1", xyz):
                raise ValueError(
                    "invalid components for 'energy_ensemble' output 'strain' "
                    "gradients: expected Labels('xyz_1', [[0], [1], [2]]) for the "
                    "first component"
                )

            if gradient.components[1] != Labels("xyz_2", xyz):
                raise ValueError(
                    "invalid components for 'energy_ensemble' output 'strain' "
                    "gradients: expected Labels('xyz_2', [[0], [1], [2]]) for the "
                    "second component"
                )

        # positions gradient checks
        if parameter == "positions":
            if gradient.samples.names != ["sample", "system", "atom"]:
                raise ValueError(
                    "invalid samples for 'energy_ensemble' output 'positions' "
                    "gradients: expected the names to be "
                    f"['sample', 'system', 'atom'], got {gradient.samples.names}"
                )

            if len(gradient.components) != 1:
                raise ValueError(
                    "invalid components for 'energy_ensemble' output 'positions' "
                    "gradients: expected one component"
                )

            if gradient.components[0] != Labels("xyz", xyz):
                raise ValueError(
                    "invalid components for 'energy_ensemble' output 'positions' "
                    "gradients: expected Labels('xyz', [[0], [1], [2]]) for the first "
                    "component"
                )
