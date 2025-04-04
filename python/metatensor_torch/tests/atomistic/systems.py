import pytest
import torch
from packaging import version

import metatensor.torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import NeighborListOptions, System

from .. import _tests_utils


@pytest.fixture
def types():
    return torch.tensor([1, -2, 3, 1, 1, 1, 3, 3])


@pytest.fixture
def positions():
    return torch.rand((8, 3))


@pytest.fixture
def cell():
    return torch.tensor([[12.0, 0, 0], [0, 12.3, 0], [0, 0, 10]])


@pytest.fixture
def pbc():
    return torch.tensor([True, True, True])


@pytest.fixture
def system(types, positions, cell, pbc):
    return System(types=types, positions=positions, cell=cell, pbc=pbc)


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


def test_system(types, positions, cell, pbc, neighbors):
    system = System(types, positions, cell, pbc)

    assert torch.all(system.types == types)
    assert torch.all(system.positions == positions)
    assert torch.all(system.cell == cell)
    assert torch.all(system.pbc == pbc)

    expected = "System with 8 atoms, periodic cell: [12, 0, 0, 0, 12.3, 0, 0, 0, 10]"
    assert str(system) == expected
    if version.parse(torch.__version__) >= version.parse("2.1"):
        # custom __repr__ definitions are only available since torch 2.1
        assert repr(system) == expected

    system = System(
        types,
        positions,
        cell=torch.zeros_like(cell),
        pbc=torch.tensor([False, False, False]),
    )
    expected = "System with 8 atoms, non periodic"
    assert str(system) == expected
    if version.parse(torch.__version__) >= version.parse("2.1"):
        # custom __repr__ definitions are only available since torch 2.1
        assert repr(system) == expected

    options = NeighborListOptions(cutoff=3.5, full_list=False, strict=True)
    system.add_neighbor_list(options, neighbors)

    assert metatensor.torch.equal_block(system.get_neighbor_list(options), neighbors)

    message = (
        "No neighbor list for NeighborListOptions\\(cutoff=3.500000, "
        "full_list=True, strict=True\\) was found.\n"
        "Is it part of the `requested_neighbor_lists` for this model?"
    )
    with pytest.raises(ValueError, match=message):
        system.get_neighbor_list(
            NeighborListOptions(cutoff=3.5, full_list=True, strict=True)
        )

    message = (
        "the neighbors list for NeighborListOptions\\(cutoff=3.500000, "
        "full_list=False, strict=True\\) already exists in this system"
    )
    with pytest.raises(ValueError, match=message):
        system.add_neighbor_list(options, neighbors)

    assert system.known_neighbor_lists() == [
        NeighborListOptions(cutoff=3.5, full_list=False, strict=True)
    ]


def test_custom_data(system):
    block = TensorBlock(
        values=torch.zeros(3, 10),
        samples=Labels.range("foo", 3),
        components=[],
        properties=Labels.range("distance", 10),
    )
    tensor = TensorMap(Labels.single(), [block])

    system.add_data("data-name", tensor)
    message = (
        "custom data 'data-name' is experimental, please contact metatensor's "
        "developers to add this data as a member of the `System` class"
    )
    with pytest.warns(UserWarning, match=message):
        stored_data = system.get_data("data-name")

    assert metatensor.torch.equal(stored_data, tensor)
    # should only warn once
    _ = system.get_data("data-name")

    assert system.known_data() == ["data-name"]

    message = (
        "custom data name 'not this' is invalid: only \\[a-z A-Z 0-9 _-\\] are accepted"
    )
    with pytest.raises(ValueError, match=message):
        system.add_data("not this", tensor)

    message = "custom data can not be named 'positions'"
    with pytest.raises(ValueError, match=message):
        system.add_data("positions", tensor)

    message = "custom data 'data-name' is already present in this system"
    with pytest.raises(ValueError, match=message):
        system.add_data("data-name", tensor)

    new_tensor = tensor.copy()
    new_tensor.block().values[:] = 12
    # this should work
    system.add_data("data-name", new_tensor, override=True)

    assert metatensor.torch.equal(system.get_data("data-name"), new_tensor)


def test_data_validation(types, positions, cell, pbc):
    # this should run without error:
    system = System(types, positions, cell, pbc)

    # ===== types checks ===== #
    message = (
        "`types`, `positions`, `cell`, and `pbc` must be on the same "
        "device, got meta, cpu, cpu, and cpu"
    )
    with pytest.raises(ValueError, match=message):
        System(types.to(device="meta"), positions, cell, pbc)

    message = (
        "new `types` must be on the same device as existing data, got meta and cpu"
    )
    with pytest.raises(ValueError, match=message):
        system.types = types.to(device="meta")

    message = "`types` must be a 1 dimensional tensor, got a tensor with 2 dimensions"
    with pytest.raises(ValueError, match=message):
        System(types.reshape(-1, 1), positions, cell, pbc)

    message = (
        "new `types` must be a 1 dimensional tensor, got a tensor with 2 dimensions"
    )
    with pytest.raises(ValueError, match=message):
        system.types = types.reshape(-1, 1)

    message = "`types` must be a tensor of integers, got torch.float64 instead"
    with pytest.raises(ValueError, match=message):
        System(types.to(dtype=torch.float64), positions, cell, pbc)

    message = "`types` must be a tensor of integers, got torch.float64 instead"
    with pytest.raises(ValueError, match=message):
        system.types = types.to(dtype=torch.float64)

    # ===== positions checks ===== #
    message = (
        "`types`, `positions`, `cell`, and `pbc` must be on the same device, "
        "got cpu, meta, cpu, and cpu"
    )
    with pytest.raises(ValueError, match=message):
        System(types, positions.to(device="meta"), cell, pbc)

    message = (
        "new `positions` must be on the same device as existing data, got meta and cpu"
    )
    with pytest.raises(ValueError, match=message):
        system.positions = positions.to(device="meta")

    message = (
        "`positions` must be a 2 dimensional tensor, got a tensor with 3 dimensions"
    )
    with pytest.raises(ValueError, match=message):
        System(types, positions.reshape(1, -1, 3), cell, pbc)

    message = (
        "new `positions` must be a 2 dimensional tensor, got a tensor with 3 dimensions"
    )
    with pytest.raises(ValueError, match=message):
        system.positions = positions.reshape(1, -1, 3)

    message = (
        "`positions` must be a \\(len\\(types\\) x 3\\) tensor, "
        "got a tensor with shape \\[8, 3\\]"
    )
    with pytest.raises(ValueError, match=message):
        System(torch.hstack([types, types]), positions, cell, pbc)

    message = (
        "`positions` must be a \\(len\\(types\\) x 3\\) tensor, "
        "got a tensor with shape \\[16, 3\\]"
    )
    with pytest.raises(ValueError, match=message):
        system.positions = torch.vstack([positions, positions])

    message = (
        "`positions` must be a tensor of floating point data, got torch.int32 instead"
    )
    with pytest.raises(ValueError, match=message):
        System(types, positions.to(dtype=torch.int32), cell, pbc)

    message = (
        "new `positions` must have the same dtype as existing data, "
        "got torch.float64 and torch.float32"
    )
    with pytest.raises(ValueError, match=message):
        system.positions = positions.to(dtype=torch.float64)

    # ===== cell checks ===== #
    message = (
        "`types`, `positions`, `cell`, and `pbc` must be on the same device, "
        "got cpu, cpu, meta, and cpu"
    )
    with pytest.raises(ValueError, match=message):
        System(types, positions, cell.to(device="meta"), pbc)

    message = "new `cell` must be on the same device as existing data, got meta and cpu"
    with pytest.raises(ValueError, match=message):
        system.cell = cell.to(device="meta")

    message = "`cell` must be a 2 dimensional tensor, got a tensor with 3 dimensions"
    with pytest.raises(ValueError, match=message):
        System(types, positions, cell.reshape(3, 1, 3), pbc)

    message = (
        "new `cell` must be a 2 dimensional tensor, got a tensor with 3 dimensions"
    )
    with pytest.raises(ValueError, match=message):
        system.cell = cell.reshape(3, 1, 3)

    message = "`cell` must be a \\(3 x 3\\) tensor, got a tensor with shape \\[6, 3\\]"
    with pytest.raises(ValueError, match=message):
        System(types, positions, torch.vstack([cell, cell]), pbc)

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
        System(types, positions, cell.to(dtype=torch.int32), pbc)

    message = (
        "new `cell` must have the same dtype as existing data, "
        "got torch.float64 and torch.float32"
    )
    with pytest.raises(ValueError, match=message):
        system.cell = cell.to(dtype=torch.float64)

    # ===== pbc checks ===== #
    message = (
        "`types`, `positions`, `cell`, and `pbc` must be on the same device, "
        "got cpu, cpu, cpu, and meta"
    )
    with pytest.raises(ValueError, match=message):
        System(types, positions, cell, pbc.to(device="meta"))

    message = "new `pbc` must be on the same device as existing data, got meta and cpu"
    with pytest.raises(ValueError, match=message):
        system.pbc = pbc.to(device="meta")

    message = "`pbc` must be a 1 dimensional tensor, got a tensor with 2 dimensions"
    with pytest.raises(ValueError, match=message):
        System(types, positions, cell, pbc.reshape(-1, 1))

    message = "new `pbc` must be a 1 dimensional tensor, got a tensor with 2 dimensions"
    with pytest.raises(ValueError, match=message):
        system.pbc = pbc.reshape(-1, 1)

    message = "`pbc` must be a tensor of booleans, got torch.int32 instead"
    with pytest.raises(ValueError, match=message):
        System(types, positions, cell, pbc.to(dtype=torch.int32))

    message = "`pbc` must be a tensor of booleans, got torch.int32 instead"
    with pytest.raises(ValueError, match=message):
        system.pbc = pbc.to(dtype=torch.int32)


def test_neighbors_validation(system):
    options = NeighborListOptions(cutoff=3.5, full_list=False, strict=True)

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
                torch.tensor([(0, 1, 0, 0)]),
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        )

        system.add_neighbor_list(options, neighbors)

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
                torch.tensor([(0, 1, 0, 0, 0)]),
            ),
            components=[Labels.range("a", 3)],
            properties=Labels.range("distance", 1),
        )

        system.add_neighbor_list(options, neighbors)

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
                torch.tensor([(0, 1, 0, 0, 0)]),
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 2),
        )

        system.add_neighbor_list(options, neighbors)

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
                torch.tensor([(0, 1, 0, 0, 0)]),
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

        system.add_neighbor_list(options, neighbors)

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
                torch.tensor([(0, 1, 0, 0, 0)]),
            ).to(device="meta"),
            components=[Labels.range("xyz", 3).to(device="meta")],
            properties=Labels.range("distance", 1).to(device="meta"),
        )

        system.add_neighbor_list(options, neighbors)

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
                torch.tensor([(0, 1, 0, 0, 0)]),
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        )

        system.add_neighbor_list(options, neighbors)


def test_to(system, neighbors):
    options = NeighborListOptions(cutoff=3.5, full_list=False, strict=True)
    system.add_neighbor_list(options, neighbors)

    block = neighbors
    system.add_data("test-data", TensorMap(Labels.single(), [block]))

    assert system.device.type == torch.device("cpu").type
    if version.parse(torch.__version__) >= version.parse("2.1"):
        check_dtype(system, torch.float32)

    converted = system.to(dtype=torch.float64)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        check_dtype(converted, torch.float64)

    devices = ["meta", torch.device("meta")]
    if _tests_utils.can_use_mps_backend():
        devices.append("mps")
        devices.append(torch.device("mps"))

    if torch.cuda.is_available():
        devices.append("cuda")
        devices.append("cuda:0")
        devices.append(torch.device("cuda"))

    for device in devices:
        moved = system.to(device=device)
        assert moved.device.type == torch.device(device).type

    # check that the code handles both positional and keyword arguments
    device = "meta"
    moved = system.to(device, dtype=torch.float32)
    moved = system.to(torch.float32, device)
    moved = system.to(torch.float32, device=device)
    moved = system.to(device, torch.float32)

    message = "can not give a device twice in `System.to`"
    with pytest.raises(ValueError, match=message):
        moved = system.to("meta", device="meta")

    message = "can not give a dtype twice in `System.to`"
    with pytest.raises(ValueError, match=message):
        moved = system.to(torch.float32, dtype=torch.float32)

    message = "unexpected type in `System.to`: Tensor"
    with pytest.raises(TypeError, match=message):
        moved = system.to(torch.tensor([0]))


# This function only works in script mode, because `block.dtype` is always an `int`, and
# `torch.dtype` is only an int in script mode.
@torch.jit.script
def check_dtype(system: System, dtype: torch.dtype):
    assert system.dtype == dtype


def test_partial_pbc():
    system = System(
        torch.tensor([1]),
        torch.tensor([[1.0, 1.0, 1.0]]),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        pbc=torch.tensor([True, False, True]),
    )

    with pytest.raises(
        ValueError,
        match="if `pbc` is False along any direction, "
        "the corresponding cell vector must be zero",
    ):
        System(
            torch.tensor([1]),
            torch.tensor([[1.0, 1.0, 1.0]]),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            pbc=torch.tensor([True, False, True]),
        )

    with pytest.raises(
        ValueError,
        match="if `pbc` is False along any direction, "
        "the corresponding cell vector must be zero",
    ):
        system.pbc = torch.tensor([True, True, False])
