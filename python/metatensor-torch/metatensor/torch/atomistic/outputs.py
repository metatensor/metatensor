from typing import Dict, List, Optional

import torch

from .. import Labels, TensorBlock, TensorMap, dtype_name
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

        if name in ["energy", "energy_ensemble"]:
            _check_energy_like(name, value, systems, request, selected_atoms)
        elif name == "features":
            _check_features(value, systems, request, selected_atoms)
        else:
            # this is a non-standard output, there is nothing to check
            continue


def _check_energy_like(
    name: str,
    value: TensorMap,
    systems: List[System],
    request: ModelOutput,
    selected_atoms: Optional[Labels],
):
    """
    Check either "energy" or "energy_ensemble" output metadata
    """
    assert name in ["energy", "energy_ensemble"]

    # Ensure the output contains a single block with the expected key
    _validate_single_block(name, value)

    # Check samples values from systems & selected_atoms
    _validate_atomic_samples(name, value, systems, request, selected_atoms)

    energy_block = value.block_by_id(0)
    device = value.device

    # Ensure that the block has no components
    _validate_no_components(name, energy_block)

    # The only difference between energy & energy_ensemble is in the properties
    if name == "energy":
        expected_properties = Labels("energy", torch.tensor([[0]], device=device))
        message = "`Labels('energy', [[0]])`"
    else:
        assert name == "energy_ensemble"
        n_ensemble_members = energy_block.values.shape[-1]
        expected_properties = Labels(
            "energy", torch.arange(n_ensemble_members, device=device).reshape(-1, 1)
        )
        message = "`Labels('energy', [[0], ..., [n]])`"

    if energy_block.properties != expected_properties:
        raise ValueError(f"invalid properties for '{name}' output: expected {message}")

    for parameter, gradient in energy_block.gradients():
        if parameter not in ["strain", "positions"]:
            raise ValueError(f"invalid gradient for '{name}' output: {parameter}")

        xyz = torch.tensor([[0], [1], [2]], device=device)
        # strain gradient checks
        if parameter == "strain":
            if gradient.samples.names != ["sample"]:
                raise ValueError(
                    f"invalid samples for '{name}' output 'strain' gradients: "
                    f"expected the names to be ['sample'], got {gradient.samples.names}"
                )

            if len(gradient.components) != 2:
                raise ValueError(
                    f"invalid components for '{name}' output 'strain' gradients: "
                    "expected two components"
                )

            if gradient.components[0] != Labels("xyz_1", xyz):
                raise ValueError(
                    f"invalid components for '{name}' output 'strain' gradients: "
                    "expected Labels('xyz_1', [[0], [1], [2]]) for the first component"
                )

            if gradient.components[1] != Labels("xyz_2", xyz):
                raise ValueError(
                    f"invalid components for '{name}' output 'strain' gradients: "
                    "expected Labels('xyz_2', [[0], [1], [2]]) for the second component"
                )

        # positions gradient checks
        if parameter == "positions":
            if gradient.samples.names != ["sample", "system", "atom"]:
                raise ValueError(
                    f"invalid samples for '{name}' output 'positions' gradients: "
                    "expected the names to be ['sample', 'system', 'atom'], "
                    f"got {gradient.samples.names}"
                )

            if len(gradient.components) != 1:
                raise ValueError(
                    f"invalid components for '{name}' output 'positions' gradients: "
                    "expected one component"
                )

            if gradient.components[0] != Labels("xyz", xyz):
                raise ValueError(
                    f"invalid components for '{name}' output 'positions' gradients: "
                    "expected Labels('xyz', [[0], [1], [2]]) for the first component"
                )


def _check_features(
    value: TensorMap,
    systems: List[System],
    request: ModelOutput,
    selected_atoms: Optional[Labels],
):
    """
    Check "features" output metadata. It is standardized with Plumed
    https://www.plumed.org/doc-master/user-doc/html/_m_e_t_a_t_e_n_s_o_r.html
    """
    # Ensure the output contains a single block with the expected key
    _validate_single_block("features", value)

    # Check samples values from systems & selected_atoms
    _validate_atomic_samples("features", value, systems, request, selected_atoms)

    features_block = value.block_by_id(0)

    # Check that the block has no components
    _validate_no_components("features", features_block)

    # Should not have any explicit gradients
    # all gradient calculations are done using autograd
    if len(features_block.gradients_list()) > 0:
        raise ValueError(
            "invalid gradients for 'features' output: it should not have any explicit",
            "gradients. all gradient calculations should be done using autograd",
        )


def _validate_atomic_samples(
    name: str,
    value: TensorMap,
    systems: List[System],
    request: ModelOutput,
    selected_atoms: Optional[Labels],
):
    """
    Validates the sample labels in the output against the expected structure
    """
    device = value.device
    block = value.block_by_id(0)

    # Check if the samples names are as expected based on whether the output is
    # per-system, per-atom, per-pair, etc.
    if request.sample_kind == ["system"]:
        expected_samples_names = ["system"]
    elif request.sample_kind == ["atom"]:
        expected_samples_names = ["system", "atom"]
    elif request.sample_kind == ["pair"]:
        expected_samples_names = ["system", "atom_1", "atom_2"]
    else:
        raise ValueError(
            f"invalid sample_kind for '{name}' output: expected one of "
            f"['system'], ['atom'], ['pair'], got {request.sample_kind}"
        )

    if block.samples.names != expected_samples_names:
        raise ValueError(
            f"invalid sample names for '{name}' output: "
            f"expected {expected_samples_names}, got {block.samples.names}"
        )

    # Check if the samples match the systems and selected_atoms
    if request.sample_kind == ["system"]:
        expected_samples = Labels(
            "system", torch.arange(len(systems), device=device).reshape(-1, 1)
        )
        if selected_atoms is not None:
            selected_systems = Labels(
                "system", torch.unique(selected_atoms.column("system")).reshape(-1, 1)
            )
            expected_samples = expected_samples.intersection(selected_systems)
        if len(expected_samples.union(block.samples)) != len(expected_samples):
            raise ValueError(
                f"invalid samples entries for '{name}' output, they do not match the "
                f"`systems` and `selected_atoms`. Expected samples:\n{expected_samples}"
            )

    elif request.sample_kind == ["atom"]:
        expected_values: List[List[int]] = []
        for s, system in enumerate(systems):
            for a in range(len(system)):
                expected_values.append([s, a])
        expected_samples = Labels(
            ["system", "atom"], torch.tensor(expected_values, device=device)
        )
        if selected_atoms is not None:
            expected_samples = expected_samples.intersection(selected_atoms)
        if len(expected_samples.union(block.samples)) != len(expected_samples):
            raise ValueError(
                f"invalid samples entries for '{name}' output, they do not match the "
                f"`systems` and `selected_atoms`. Expected samples:\n{expected_samples}"
            )

    elif request.sample_kind == ["pair"]:
        # this one is a bit more complicated, because not all pairs might be present
        # due to the cutoff of the neighbor list. The condition that we check here
        # is that there are no extra [system, atom] pairs in the output, where `atom`
        # will be taken to be atom_1 and atom_2 in turn.

        # get all allowed [system, atom] pairs
        allowed_values: List[List[int]] = []
        for s, system in enumerate(systems):
            for a in range(len(system)):
                allowed_values.append([s, a])
        allowed_samples = Labels(
            ["system", "atom"], torch.tensor(allowed_values, device=device)
        )
        if selected_atoms is not None:
            allowed_samples = allowed_samples.intersection(selected_atoms)
        allowed_sys_atom = allowed_samples.values[:, [0, 1]]
        allowed_sys_atom_list: List[List[int]] = allowed_sys_atom.tolist()

        # check [system, atom_1]
        actual_sys_atom_1 = block.samples.values[:, [0, 1]]
        actual_sys_atom_1_list: List[List[int]] = actual_sys_atom_1.tolist()
        for sys_and_atom in actual_sys_atom_1_list:
            if sys_and_atom not in allowed_sys_atom_list:
                raise ValueError(
                    f"invalid samples entries for '{name}' output: "
                    f"unexpected [system, atom_1] pair {sys_and_atom} "
                    "in the output"
                )

        # check [system, atom_2]
        actual_sys_atom_2 = block.samples.values[:, [0, 2]]
        actual_sys_atom_2_list: List[List[int]] = actual_sys_atom_2.tolist()
        for sys_and_atom in actual_sys_atom_2_list:
            if sys_and_atom not in allowed_sys_atom_list:
                raise ValueError(
                    f"invalid samples entries for '{name}' output: "
                    f"unexpected [system, atom_2] pair {sys_and_atom} "
                    "in the output"
                )

    else:
        raise ValueError(
            f"invalid sample_kind for '{name}' output: expected one of "
            f"['system'], ['atom'], ['pair'], got {request.sample_kind}"
        )


def _validate_single_block(name: str, value: TensorMap):
    """
    Ensure the TensorMap has a single block with the expected key
    """
    if value.keys != Labels("_", torch.tensor([[0]])):
        raise ValueError(
            f"invalid keys for '{name}' output: expected `Labels('_', [[0]])`"
        )


def _validate_no_components(name: str, block: TensorBlock):
    """
    Ensure the block has no components"
    """
    if len(block.components) != 0:
        raise ValueError(
            f"invalid components for {name} output: components should be empty"
        )
