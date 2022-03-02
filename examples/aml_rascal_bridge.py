from typing import List
import copy
import numpy as np
import torch

import ase
from rascal.representations import SphericalExpansion

from aml_storage import Labels, Block, Descriptor


class RascalSphericalExpansion:
    def __init__(self, hypers):
        self._hypers = copy.deepcopy(hypers)

    def compute(self, frames: List[ase.Atoms]) -> Descriptor:
        # Step 1: compute spherical expansion with librascal
        hypers = copy.deepcopy(self._hypers)
        global_species = list(
            map(int, np.unique(np.concatenate([f.numbers for f in frames])))
        )

        hypers["global_species"] = global_species
        hypers["expansion_by_species_method"] = "user defined"

        calculator = SphericalExpansion(**hypers)
        manager = calculator.transform(frames)
        data = manager.get_features(calculator)
        gradients = manager.get_features_gradient(calculator)
        info = manager.get_representation_info()
        grad_info = manager.get_gradients_info()

        # Step 2: move data around to follow the storage convention
        sparse = Labels(
            names=["spherical_harmonics_l", "center_species", "neighbor_species"],
            values=np.array(
                [
                    [l, center_species, neighbor_species]
                    for l in range(hypers["max_angular"] + 1)
                    for center_species in global_species
                    for neighbor_species in global_species
                ],
                dtype=np.int32,
            ),
        )

        features = Labels(
            names=["n"],
            values=np.array([[n] for n in range(hypers["max_radial"])], dtype=np.int32),
        )

        lm_slices = []
        start = 0
        for l in range(hypers["max_angular"] + 1):
            stop = start + 2 * l + 1
            lm_slices.append(slice(start, stop))
            start = stop

        data = data.reshape(
            data.shape[0], len(global_species), hypers["max_radial"], -1
        )
        gradients = gradients.reshape(
            gradients.shape[0], len(global_species), hypers["max_radial"], -1
        )

        global_to_per_structure_atom_id = []
        for frame in frames:
            for i in range(len(frame)):
                global_to_per_structure_atom_id.append(i)

        blocks = []
        for sparse_i, (l, center_species, neighbor_species) in enumerate(sparse):
            neighbor_species_i = global_species.index(neighbor_species)
            center_species_mask = np.where(info[:, 2] == center_species)[0]
            block_data = data[center_species_mask, neighbor_species_i, :, lm_slices[l]]
            block_data = block_data.swapaxes(1, 2)

            samples = Labels(
                names=["structure", "center"],
                values=np.copy(info[center_species_mask, :2]).astype(np.int32),
            )
            symmetric = Labels(
                names=["spherical_harmonics_m"],
                values=np.array([[m] for m in range(-l, l + 1)], dtype=np.int32),
            )

            block_gradients = None
            gradient_samples = None
            if hypers["compute_gradients"]:
                gradient_samples = []
                block_gradients = []
                for sample_i, (structure, center) in enumerate(samples):
                    gradient_mask = np.apply_along_axis(
                        lambda row: row[0] == structure
                        and row[1] == center
                        and row[4] == neighbor_species,
                        axis=1,
                        arr=grad_info,
                    )

                    for grad_index in np.where(gradient_mask)[0]:
                        start = 3 * grad_index
                        stop = 3 * grad_index + 3
                        block_gradient = gradients[
                            start:stop, neighbor_species_i, :, lm_slices[l]
                        ]
                        block_gradient = block_gradient.swapaxes(1, 2)
                        if np.linalg.norm(block_gradient) == 0:
                            continue

                        block_gradients.append(block_gradient)

                        neighbor = global_to_per_structure_atom_id[
                            grad_info[grad_index, 2]
                        ]
                        gradient_samples.append((sample_i, neighbor, 0))
                        gradient_samples.append((sample_i, neighbor, 1))
                        gradient_samples.append((sample_i, neighbor, 2))

                if len(gradient_samples) != 0:
                    block_gradients = np.concatenate(block_gradients)
                    gradient_samples = Labels(
                        names=["sample", "atom", "spatial"],
                        values=np.vstack(gradient_samples).astype(np.int32),
                    )
                else:
                    block_gradients = np.zeros(
                        (0, symmetric.shape[0], features.shape[0])
                    )
                    gradient_samples = Labels(
                        names=["sample", "atom", "spatial"],
                        values=np.zeros((0, 3), dtype=np.int32),
                    )

            # reset atom index (librascal uses a global atom index)
            samples = Labels(
                names=["structure", "center"],
                values=np.array(
                    [
                        (structure, global_to_per_structure_atom_id[center])
                        for structure, center in samples
                    ],
                    dtype=np.int32,
                ),
            )

            block = Block(
                data=np.copy(block_data),
                samples=samples,
                symmetric=symmetric,
                features=features,
            )

            if block_gradients is not None:
                block.add_gradient("positions", gradient_samples, block_gradients)

            blocks.append(block)

        return Descriptor(sparse, blocks)


class SphericalExpansionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, calculator, positions, cell, numbers):
        assert isinstance(positions, torch.Tensor)
        assert isinstance(cell, torch.Tensor)
        assert isinstance(numbers, torch.Tensor)

        frame = ase.Atoms(
            numbers=numbers.numpy(), cell=cell.numpy(), positions=positions.numpy()
        )
        manager = calculator.transform(frame)
        descriptor = manager.get_features(calculator)

        samples = manager.get_representation_info()

        grad_descriptor = manager.get_features_gradient(calculator)
        grad_info = manager.get_gradients_info()
        ctx.save_for_backward(
            torch.tensor(grad_descriptor),
            torch.tensor(grad_info),
        )

        return samples, torch.tensor(descriptor)

    @staticmethod
    def backward(ctx, grad_samples, grad_output):
        grad_spherical_expansion, grad_info = ctx.saved_tensors

        grad_calculator = grad_positions = grad_cell = grad_numbers = None

        if ctx.needs_input_grad[0]:
            raise ValueError("can not compute gradients w.r.t. calculator")
        if ctx.needs_input_grad[1]:
            if grad_spherical_expansion.shape[1] == 0:
                raise ValueError(
                    "missing gradients, please set `compute_gradients` to True"
                )

            grad_positions = torch.zeros((grad_output.shape[0], 3))
            for sample, (_, center_i, neighbor_i, *_) in enumerate(grad_info):
                for spatial_i in range(3):
                    sample_i = 3 * sample + spatial_i
                    grad_positions[neighbor_i, spatial_i] += torch.dot(
                        grad_output[center_i, :], grad_spherical_expansion[sample_i, :]
                    )

        if ctx.needs_input_grad[2]:
            raise ValueError("can not compute gradients w.r.t. cell")
        if ctx.needs_input_grad[2]:
            raise ValueError("can not compute gradients w.r.t. atomic numbers")

        return grad_calculator, grad_positions, grad_cell, grad_numbers


class TorchFrame:
    def __init__(self, frame: ase.Atoms, requires_grad: bool):
        self.positions = torch.tensor(frame.positions, requires_grad=requires_grad)
        self.species = torch.tensor(frame.numbers)
        self.cell = torch.tensor(frame.cell[:])

    def __len__(self):
        return self.species.shape[0]


class RascalSphericalExpansionTorch:
    def __init__(self, hypers):
        self._hypers = copy.deepcopy(hypers)

    def compute(self, frames: List[TorchFrame]) -> Descriptor:
        # Step 1: compute spherical expansion with librascal
        hypers = copy.deepcopy(self._hypers)
        global_species = list(
            map(int, np.unique(np.concatenate([f.species.numpy() for f in frames])))
        )

        hypers["global_species"] = global_species
        hypers["expansion_by_species_method"] = "user defined"

        calculator = SphericalExpansion(**hypers)

        all_spx = []
        all_info = []
        for i, frame in enumerate(frames):
            info, spx = SphericalExpansionAutograd.apply(
                calculator, frame.positions, frame.cell, frame.species
            )

            all_spx.append(spx)

            # set the structure id
            info[:, 0] = i
            all_info.append(info)

        data = torch.vstack(all_spx)
        info = np.vstack(all_info)

        # Step 2: move data around to follow the storage convention
        sparse = Labels(
            names=["spherical_harmonics_l", "center_species", "neighbor_species"],
            values=np.array(
                [
                    [l, center_species, neighbor_species]
                    for l in range(hypers["max_angular"] + 1)
                    for center_species in global_species
                    for neighbor_species in global_species
                ],
                dtype=np.int32,
            ),
        )

        features = Labels(
            names=["n"],
            values=np.array([[n] for n in range(hypers["max_radial"])], dtype=np.int32),
        )

        lm_slices = []
        start = 0
        for l in range(hypers["max_angular"] + 1):
            stop = start + 2 * l + 1
            lm_slices.append(slice(start, stop))
            start = stop

        data = data.reshape(
            data.shape[0], len(global_species), hypers["max_radial"], -1
        )

        blocks = []
        for sparse_i, (l, center_species, neighbor_species) in enumerate(sparse):
            neighbor_species_i = global_species.index(neighbor_species)
            center_species_mask = np.where(info[:, 2] == center_species)[0]
            block_data = data[center_species_mask, neighbor_species_i, :, lm_slices[l]]
            block_data = block_data.swapaxes(1, 2)

            samples = Labels(
                names=["structure", "center"],
                values=np.copy(info[center_species_mask, :2]).astype(np.int32),
            )
            symmetric = Labels(
                names=["spherical_harmonics_m"],
                values=np.array([[m] for m in range(-l, l + 1)], dtype=np.int32),
            )

            blocks.append(
                Block(
                    data=block_data,
                    samples=samples,
                    symmetric=symmetric,
                    features=features,
                )
            )

        return Descriptor(sparse, blocks)
