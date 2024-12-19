import io
import os
import warnings

import pytest
import torch

import metatensor.torch
import metatensor.torch.atomistic
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import NeighborListOptions, System


@pytest.mark.parametrize("buffer", [True, False])
def test_save_load(tmpdir, buffer):
    system = System(
        types=torch.tensor([1, 2, 3, 4]),
        positions=torch.rand((4, 3), dtype=torch.float64),
        cell=torch.rand((3, 3), dtype=torch.float64),
        pbc=torch.tensor([True, True, True], dtype=torch.bool),
    )
    nl_block = TensorBlock(
        values=torch.rand(100, 3, 1, dtype=torch.float64),
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            torch.arange(100 * 5, dtype=torch.int64).reshape(100, 5),
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
    system.add_neighbor_list(
        NeighborListOptions(cutoff=3.5, full_list=True, strict=True),
        nl_block,
    )
    system.add_neighbor_list(
        NeighborListOptions(cutoff=4.0, full_list=False, strict=False), nl_block
    )

    system.add_data(
        "my_data",
        TensorMap(
            Labels.single(),
            [metatensor.torch.block_from_array(torch.rand(10, 8, dtype=torch.float64))],
        ),
    )
    system.add_data(
        "more_data",
        TensorMap(
            Labels.single(),
            [metatensor.torch.block_from_array(torch.rand(22, 1, dtype=torch.float64))],
        ),
    )

    if buffer:
        path_or_buffer = io.BytesIO()
    else:
        path_or_buffer = os.path.join(tmpdir, "system.mta")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "custom data '.*' is experimental")
        metatensor.torch.atomistic.save(path_or_buffer, system)

    system_loaded = metatensor.torch.atomistic.load_system(path_or_buffer)
    assert torch.equal(system.types, system_loaded.types)
    assert torch.equal(system.positions, system_loaded.positions)
    assert torch.equal(system.cell, system_loaded.cell)
    assert torch.equal(system.pbc, system_loaded.pbc)

    neigbor_list_1 = system.get_neighbor_list(
        NeighborListOptions(cutoff=3.5, full_list=True, strict=True)
    )
    neighbor_list_1_loaded = system_loaded.get_neighbor_list(
        NeighborListOptions(cutoff=3.5, full_list=True, strict=True)
    )
    neighbor_list_2 = system.get_neighbor_list(
        NeighborListOptions(cutoff=4.0, full_list=False, strict=False)
    )
    neighbor_list_2_loaded = system_loaded.get_neighbor_list(
        NeighborListOptions(cutoff=4.0, full_list=False, strict=False)
    )
    assert metatensor.torch.equal_block(neigbor_list_1, neighbor_list_1_loaded)
    assert metatensor.torch.equal_block(neighbor_list_2, neighbor_list_2_loaded)

    assert set(system.known_data()) == set(system_loaded.known_data())
    assert metatensor.torch.equal(
        system.get_data("my_data"), system_loaded.get_data("my_data")
    )
    assert metatensor.torch.equal(
        system.get_data("more_data"), system_loaded.get_data("more_data")
    )
