import os

import pytest
import torch

import metatensor.torch
from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import NeighborListOptions, System, load, save


@pytest.mark.parametrize(
    "pbc", [[True, True, True], [True, False, True], [False, False, False]]
)
@pytest.mark.parametrize("nl_size", [10, 100, 1000])
@pytest.mark.parametrize("cutoff", [3.5, 5.0])
@pytest.mark.parametrize("full_list", [True, False])
@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("extra_data", [False, True])
@pytest.mark.filterwarnings("ignore:custom data")
def test_save_load(tmpdir, pbc, cutoff, nl_size, full_list, strict, extra_data):
    cell = torch.rand((3, 3), dtype=torch.float64)
    cell[[not periodic for periodic in pbc]] = 0.0

    system = System(
        types=torch.tensor([1, 2, 3, 4]),
        positions=torch.rand((4, 3), dtype=torch.float64),
        cell=cell,
        pbc=torch.tensor(pbc, dtype=torch.bool),
    )
    system.add_neighbor_list(
        NeighborListOptions(cutoff=cutoff, full_list=full_list, strict=strict),
        TensorBlock(
            values=torch.rand(nl_size, 3, 1, dtype=torch.float64),
            samples=Labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                torch.arange(nl_size * 5, dtype=torch.int64).reshape(nl_size, 5),
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        ),
    )

    if extra_data:
        my_data = metatensor.torch.block_from_array(
            torch.rand(10, 8, dtype=torch.float64)
        )
        system.add_data("my_data", my_data)

    save(os.path.join(tmpdir, "system.npz"), system)
    system_loaded = load(os.path.join(tmpdir, "system.npz"))
    assert torch.equal(system.types, system_loaded.types)
    assert torch.equal(system.positions, system_loaded.positions)
    assert torch.equal(system.cell, system_loaded.cell)
    assert torch.equal(system.pbc, system_loaded.pbc)
    neigbor_list = system.get_neighbor_list(
        NeighborListOptions(cutoff=cutoff, full_list=full_list, strict=strict)
    )
    neighbor_list_loaded = system_loaded.get_neighbor_list(
        NeighborListOptions(cutoff=cutoff, full_list=full_list, strict=strict)
    )
    assert metatensor.torch.equal_block(neigbor_list, neighbor_list_loaded)

    if extra_data:
        assert system.known_data() == system_loaded.known_data()
        assert metatensor.torch.equal_block(
            system.get_data("my_data"), system_loaded.get_data("my_data")
        )
