import pytest
import torch

import metatensor.torch
from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import NeighborsListOptions, System


@pytest.fixture
def positions():
    return TensorBlock(
        values=torch.zeros(8, 3, 1),
        samples=Labels(
            ["atom", "species"],
            torch.tensor(
                [
                    (0, 1),
                    (1, -2),
                    (2, 3),
                    (3, 1),
                    (7, 1),
                    (8, 1),
                    (9, 3),
                    (11, 3),
                ]
            ),
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("position", 1),
    )


@pytest.fixture
def cell():
    return TensorBlock(
        values=torch.zeros(1, 3, 3, 1),
        samples=Labels.range("_", 1),
        components=[Labels.range("cell_abc", 3), Labels.range("xyz", 3)],
        properties=Labels.range("cell", 1),
    )


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


def test_system(positions, cell, neighbors):
    system = System(positions=positions, cell=cell)

    assert metatensor.torch.equal_block(system.positions, positions)
    assert metatensor.torch.equal_block(system.cell, cell)

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

    data = TensorBlock(
        values=torch.zeros(3, 10),
        samples=Labels.range("foo", 3),
        components=[],
        properties=Labels.range("distance", 10),
    )

    system.add_data("data name", data)
    message = (
        "custom data \\(data name\\) is experimental, "
        "please contact the developers to add your data in the main API"
    )
    with pytest.warns(UserWarning, match=message):
        stored_data = system.get_data("data name")

    assert metatensor.torch.equal_block(stored_data, data)
    # should only warn once
    _ = system.get_data("data name")

    assert system.known_data() == ["data name"]

    assert metatensor.torch.equal_block(system.get_data("positions"), positions)
    assert metatensor.torch.equal_block(system.get_data("cell"), cell)

    message = "custom data can not be 'positions', 'cell', or 'neighbors'"
    with pytest.raises(ValueError, match=message):
        system.add_data("positions", data)

    message = "custom data for 'data name' is already present in this system"
    with pytest.raises(ValueError, match=message):
        system.add_data("data name", data)


def test_data_validation(positions, cell):
    message = (
        "invalid samples for `positions`: "
        "the samples names must be 'atom' and 'species'"
    )
    with pytest.raises(ValueError, match=message):
        bad_positions = TensorBlock(
            values=torch.zeros(1, 3, 1),
            samples=Labels(["nope", "species"], torch.tensor([(0, 1)])),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("position", 1),
        )
        System(positions=bad_positions, cell=cell)

    message = (
        "invalid components for `positions`: "
        "there should be a single 'xyz'=\\[0, 1, 2\\] component"
    )
    with pytest.raises(ValueError, match=message):
        bad_positions = TensorBlock(
            values=torch.zeros(1, 1),
            samples=Labels(["atom", "species"], torch.tensor([(0, 1)])),
            components=[],
            properties=Labels.range("position", 1),
        )
        System(positions=bad_positions, cell=cell)

    message = (
        "invalid properties for `positions`: "
        "there should be a single 'positions'=0 property"
    )
    with pytest.raises(ValueError, match=message):
        bad_positions = TensorBlock(
            values=torch.zeros(1, 3, 1),
            samples=Labels(["atom", "species"], torch.tensor([(0, 1)])),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("hey", 1),
        )
        System(positions=bad_positions, cell=cell)

    message = "`positions` should not have any gradients"
    with pytest.raises(ValueError, match=message):
        bad_positions = TensorBlock(
            values=torch.zeros(1, 3, 1),
            samples=Labels(["atom", "species"], torch.tensor([(0, 1)])),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("position", 1),
        )
        bad_positions.add_gradient(
            "gradient",
            TensorBlock(
                values=torch.zeros(1, 3, 1),
                samples=Labels.range("sample", 1),
                components=[Labels.range("xyz", 3)],
                properties=Labels.range("position", 1),
            ),
        )

        System(positions=bad_positions, cell=cell)

    message = "invalid samples for `cell`: there should be a single '_'=0 sample"
    with pytest.raises(ValueError, match=message):
        bad_cell = TensorBlock(
            values=torch.zeros(1, 3, 3, 1),
            samples=Labels.range("nope", 1),
            components=[Labels.range("cell_abc", 3), Labels.range("xyz", 3)],
            properties=Labels.range("cell", 1),
        )
        System(positions=positions, cell=bad_cell)

    message = "invalid components for `cell`: there should be 2 components, got 0"
    with pytest.raises(ValueError, match=message):
        bad_cell = TensorBlock(
            values=torch.zeros(1, 1),
            samples=Labels.range("_", 1),
            components=[],
            properties=Labels.range("cell", 1),
        )
        System(positions=positions, cell=bad_cell)

    message = (
        "invalid components for `cell`: "
        "the first component should be 'cell_abc'=\\[0, 1, 2\\]"
    )
    with pytest.raises(ValueError, match=message):
        bad_cell = TensorBlock(
            values=torch.zeros(1, 3, 3, 1),
            samples=Labels.range("_", 1),
            components=[Labels.range("abc", 3), Labels.range("xyz", 3)],
            properties=Labels.range("cell", 1),
        )
        System(positions=positions, cell=bad_cell)

    message = (
        "invalid components for `cell`: "
        "the second component should be 'xyz'=\\[0, 1, 2\\]"
    )
    with pytest.raises(ValueError, match=message):
        bad_cell = TensorBlock(
            values=torch.zeros(1, 3, 3, 1),
            samples=Labels.range("_", 1),
            components=[Labels.range("cell_abc", 3), Labels.range("xyz_def", 3)],
            properties=Labels.range("cell", 1),
        )
        System(positions=positions, cell=bad_cell)

    message = (
        "invalid properties for `cell`: there should be a single 'cell'=0 property"
    )
    with pytest.raises(ValueError, match=message):
        bad_cell = TensorBlock(
            values=torch.zeros(1, 3, 3, 1),
            samples=Labels.range("_", 1),
            components=[Labels.range("cell_abc", 3), Labels.range("xyz", 3)],
            properties=Labels.range("foo", 1),
        )
        System(positions=positions, cell=bad_cell)

    message = "`cell` should not have any gradients"
    with pytest.raises(ValueError, match=message):
        bad_cell = TensorBlock(
            values=torch.zeros(1, 3, 3, 1),
            samples=Labels.range("_", 1),
            components=[Labels.range("cell_abc", 3), Labels.range("xyz", 3)],
            properties=Labels.range("cell", 1),
        )
        bad_cell.add_gradient(
            "gradient",
            TensorBlock(
                values=torch.zeros(1, 3, 3, 1),
                samples=Labels.range("sample", 1),
                components=[Labels.range("cell_abc", 3), Labels.range("xyz", 3)],
                properties=Labels.range("cell", 1),
            ),
        )

        System(positions=positions, cell=bad_cell)


def test_neighbors_validation(positions, cell):
    system = System(positions=positions, cell=cell)
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
