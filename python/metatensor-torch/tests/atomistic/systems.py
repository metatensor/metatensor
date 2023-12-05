import pytest
import torch

import metatensor.torch
from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import NeighborsListOptions, System


@pytest.fixture
def species():
    return torch.IntTensor([1, -2, 3, 1, 1, 1, 3, 3])


@pytest.fixture
def positions():
    return torch.rand((8, 3))


@pytest.fixture
def cell():
    return torch.rand((3, 3))


@pytest.fixture
def system(species, positions, cell):
    return System(species=species, positions=positions, cell=cell)


@pytest.fixture
def neighbors():
    return TensorBlock(
        values=torch.zeros(2, 3, 1),
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            torch.tensor(
                [
                    (0, 1, 0, 0, 0),
                    (0, 2, 1, 0, -1),
                ]
            ),
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )


def test_system(species, positions, cell, neighbors):
    system = System(species, positions, cell)

    assert torch.all(system.species == species)
    assert torch.all(system.positions == positions)
    assert torch.all(system.cell == cell)

    options = NeighborsListOptions(model_cutoff=3.5, full_list=False)
    system.add_neighbors_list(options, neighbors)

    assert metatensor.torch.equal_block(system.get_neighbors_list(options), neighbors)

    message = (
        "No neighbors list for NeighborsListOptions\\(cutoff=3.500000, "
        "full_list=True\\) was found.\n"
        "Is it part of the `requested_neighbors_lists` for this model?"
    )
    with pytest.raises(ValueError, match=message):
        system.get_neighbors_list(
            NeighborsListOptions(model_cutoff=3.5, full_list=True)
        )

    message = (
        "the neighbors list for NeighborsListOptions\\(cutoff=3.500000, "
        "full_list=False\\) already exists in this system"
    )
    with pytest.raises(ValueError, match=message):
        system.add_neighbors_list(options, neighbors)

    assert system.known_neighbors_lists() == [
        NeighborsListOptions(model_cutoff=3.5, full_list=False)
    ]


def test_custom_data(system):
    data = TensorBlock(
        values=torch.zeros(3, 10),
        samples=Labels.range("foo", 3),
        components=[],
        properties=Labels.range("distance", 10),
    )

    system.add_data("data-name", data)
    message = (
        "custom data 'data-name' is experimental, please contact metatensor's "
        "developers to add this data as a member of the `System` class"
    )
    with pytest.warns(UserWarning, match=message):
        stored_data = system.get_data("data-name")

    assert metatensor.torch.equal_block(stored_data, data)
    # should only warn once
    _ = system.get_data("data-name")

    assert system.known_data() == ["data-name"]

    message = (
        "custom data name 'not this' is invalid: only \\[a-z A-Z 0-9 _-\\] are accepted"
    )
    with pytest.raises(ValueError, match=message):
        system.add_data("not this", data)

    message = "custom data can not be named 'positions'"
    with pytest.raises(ValueError, match=message):
        system.add_data("positions", data)

    message = "custom data 'data-name' is already present in this system"
    with pytest.raises(ValueError, match=message):
        system.add_data("data-name", data)


def test_data_validation(species, positions, cell):
    # this should run without error:
    system = System(species, positions, cell)

    # ===== species checks ===== #
    message = (
        "`species`, `positions`, and `cell` must be on the same device, "
        "got meta, cpu, and cpu"
    )
    with pytest.raises(ValueError, match=message):
        System(species.to(device="meta"), positions, cell)

    message = (
        "new `species` must be on the same device as existing data, got meta and cpu"
    )
    with pytest.raises(ValueError, match=message):
        system.species = species.to(device="meta")

    message = "`species` must be a 1 dimensional tensor, got a tensor with 2 dimensions"
    with pytest.raises(ValueError, match=message):
        System(species.reshape(-1, 1), positions, cell)

    message = (
        "new `species` must be a 1 dimensional tensor, got a tensor with 2 dimensions"
    )
    with pytest.raises(ValueError, match=message):
        system.species = species.reshape(-1, 1)

    message = "`species` must be a tensor of integers, got torch.float64 instead"
    with pytest.raises(ValueError, match=message):
        System(species.to(dtype=torch.float64), positions, cell)

    message = "`species` must be a tensor of integers, got torch.float64 instead"
    with pytest.raises(ValueError, match=message):
        system.species = species.to(dtype=torch.float64)

    # ===== positions checks ===== #
    message = (
        "`species`, `positions`, and `cell` must be on the same device, "
        "got cpu, meta, and cpu"
    )
    with pytest.raises(ValueError, match=message):
        System(species, positions.to(device="meta"), cell)

    message = (
        "new `positions` must be on the same device as existing data, got meta and cpu"
    )
    with pytest.raises(ValueError, match=message):
        system.positions = positions.to(device="meta")

    message = (
        "`positions` must be a 2 dimensional tensor, got a tensor with 3 dimensions"
    )
    with pytest.raises(ValueError, match=message):
        System(species, positions.reshape(1, -1, 3), cell)

    message = (
        "new `positions` must be a 2 dimensional tensor, got a tensor with 3 dimensions"
    )
    with pytest.raises(ValueError, match=message):
        system.positions = positions.reshape(1, -1, 3)

    message = (
        "`positions` must be a \\(n_atoms x 3\\) tensor, "
        "got a tensor with shape \\[8, 3\\]"
    )
    with pytest.raises(ValueError, match=message):
        System(torch.hstack([species, species]), positions, cell)

    message = (
        "`positions` must be a \\(n_atoms x 3\\) tensor, "
        "got a tensor with shape \\[16, 3\\]"
    )
    with pytest.raises(ValueError, match=message):
        system.positions = torch.vstack([positions, positions])

    message = (
        "`positions` must be a tensor of floating point data, got torch.int32 instead"
    )
    with pytest.raises(ValueError, match=message):
        System(species, positions.to(dtype=torch.int32), cell)

    message = (
        "new `positions` must have the same dtype as existing data, "
        "got torch.float64 and torch.float32"
    )
    with pytest.raises(ValueError, match=message):
        system.positions = positions.to(dtype=torch.float64)

    # ===== cell checks ===== #
    message = (
        "`species`, `positions`, and `cell` must be on the same device, "
        "got cpu, cpu, and meta"
    )
    with pytest.raises(ValueError, match=message):
        System(species, positions, cell.to(device="meta"))

    message = "new `cell` must be on the same device as existing data, got meta and cpu"
    with pytest.raises(ValueError, match=message):
        system.cell = cell.to(device="meta")

    message = "`cell` must be a 2 dimensional tensor, got a tensor with 3 dimensions"
    with pytest.raises(ValueError, match=message):
        System(species, positions, cell.reshape(3, 1, 3))

    message = (
        "new `cell` must be a 2 dimensional tensor, got a tensor with 3 dimensions"
    )
    with pytest.raises(ValueError, match=message):
        system.cell = cell.reshape(3, 1, 3)

    message = "`cell` must be a \\(3 x 3\\) tensor, got a tensor with shape \\[6, 3\\]"
    with pytest.raises(ValueError, match=message):
        System(species, positions, torch.vstack([cell, cell]))

    message = (
        "new `cell` must be a \\(3 x 3\\) tensor, got a tensor with shape \\[6, 3\\]"
    )
    with pytest.raises(ValueError, match=message):
        system.cell = torch.vstack([cell, cell])

    message = (
        "`cell` must be have the same dtype as `positions`, "
        "got torch.int32 and torch.float32"
    )
    with pytest.raises(ValueError, match=message):
        System(species, positions, cell.to(dtype=torch.int32))

    message = (
        "new `cell` must have the same dtype as existing data, "
        "got torch.float64 and torch.float32"
    )
    with pytest.raises(ValueError, match=message):
        system.cell = cell.to(dtype=torch.float64)


def test_neighbors_validation(system):
    options = NeighborsListOptions(model_cutoff=3.5, full_list=False)

    message = (
        "invalid samples for `neighbors`: the samples names must be 'first_atom', "
        "'second_atom', 'cell_shift_a', 'cell_shift_b', 'cell_shift_c'"
    )
    with pytest.raises(ValueError, match=message):
        neighbors = TensorBlock(
            values=torch.zeros(1, 3, 1),
            samples=Labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                ],
                torch.tensor(
                    [
                        (0, 1, 0, 0),
                    ]
                ),
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        )

        system.add_neighbors_list(options, neighbors)

    message = (
        "invalid components for `neighbors`: "
        "there should be a single 'xyz'=\\[0, 1, 2\\] component"
    )
    with pytest.raises(ValueError, match=message):
        neighbors = TensorBlock(
            values=torch.zeros(1, 3, 1),
            samples=Labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                torch.tensor(
                    [
                        (0, 1, 0, 0, 0),
                    ]
                ),
            ),
            components=[Labels.range("a", 3)],
            properties=Labels.range("distance", 1),
        )

        system.add_neighbors_list(options, neighbors)

    message = (
        "invalid properties for `neighbors`: "
        "there should be a single 'distance'=0 property"
    )
    with pytest.raises(ValueError, match=message):
        neighbors = TensorBlock(
            values=torch.zeros(1, 3, 2),
            samples=Labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                torch.tensor(
                    [
                        (0, 1, 0, 0, 0),
                    ]
                ),
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 2),
        )

        system.add_neighbors_list(options, neighbors)

    message = "`neighbors` should not have any gradients"
    with pytest.raises(ValueError, match=message):
        neighbors = TensorBlock(
            values=torch.zeros(1, 3, 1),
            samples=Labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                torch.tensor(
                    [
                        (0, 1, 0, 0, 0),
                    ]
                ),
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        )

        neighbors.add_gradient(
            "gradient",
            TensorBlock(
                values=torch.zeros(1, 3, 1),
                samples=Labels.range("sample", 1),
                components=[Labels.range("xyz", 3)],
                properties=Labels.range("distance", 1),
            ),
        )

        system.add_neighbors_list(options, neighbors)

    message = (
        "`neighbors` device \\(meta\\) does not match this system's device \\(cpu\\)"
    )
    with pytest.raises(ValueError, match=message):
        neighbors = TensorBlock(
            values=torch.zeros(1, 3, 1).to(device="meta"),
            samples=Labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                torch.tensor(
                    [
                        (0, 1, 0, 0, 0),
                    ]
                ),
            ).to(device="meta"),
            components=[Labels.range("xyz", 3).to(device="meta")],
            properties=Labels.range("distance", 1).to(device="meta"),
        )

        system.add_neighbors_list(options, neighbors)

    message = (
        "`neighbors` dtype \\(torch.float64\\) does not match "
        "this system's dtype \\(torch.float32\\)"
    )
    with pytest.raises(ValueError, match=message):
        neighbors = TensorBlock(
            values=torch.zeros(1, 3, 1).to(dtype=torch.float64),
            samples=Labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                torch.tensor(
                    [
                        (0, 1, 0, 0, 0),
                    ]
                ),
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        )

        system.add_neighbors_list(options, neighbors)
