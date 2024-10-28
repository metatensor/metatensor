import pytest
import torch
from packaging import version

from metatensor.torch.atomistic import (
    NeighborListOptions,
    System,
    register_autograd_neighbors,
)


try:
    import ase

    from metatensor.torch.atomistic.ase_calculator import _compute_ase_neighbors

    HAVE_ASE = True
except ImportError:
    HAVE_ASE = False


def test_neighbor_list_options():
    options = NeighborListOptions(3.4, True, "hello")

    assert options.cutoff == 3.4
    assert options.full_list
    assert options.requestors() == ["hello"]

    options.add_requestor("another one")
    assert options.requestors() == ["hello", "another one"]

    # No empty requestors, no duplicated requestors
    options.add_requestor("")
    options.add_requestor("hello")
    assert options.requestors() == ["hello", "another one"]

    assert NeighborListOptions(3.4, True, "a") == NeighborListOptions(3.4, True, "b")
    assert NeighborListOptions(3.4, True) != NeighborListOptions(3.4, False)
    assert NeighborListOptions(3.4, True) != NeighborListOptions(3.5, True)

    expected = "NeighborListOptions(cutoff=3.400000, full_list=True)"
    assert str(options) == expected

    expected = """NeighborListOptions
    cutoff: 3.400000
    full_list: True
    requested by:
        - hello
        - another one
"""
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert repr(options) == expected


@pytest.mark.skipif(not HAVE_ASE, reason="this tests requires ASE neighbor list")
def test_neighbors_autograd():
    torch.manual_seed(0xDEADBEEF)
    n_atoms = 20
    approx_cell_size = 6.0
    positions = approx_cell_size * torch.rand(
        n_atoms, 3, dtype=torch.float64, requires_grad=True
    )
    cell = approx_cell_size * (
        torch.eye(3, dtype=torch.float64) + 0.1 * torch.rand(3, 3, dtype=torch.float64)
    )
    cell.requires_grad = True

    def compute(positions, cell, options):
        atoms = ase.Atoms(
            "C" * positions.shape[0],
            positions=positions.detach().numpy(),
            cell=cell.detach().numpy(),
            pbc=True,
        )
        neighbors = _compute_ase_neighbors(
            atoms, options, dtype=torch.float64, device="cpu"
        )

        system = System(
            torch.from_numpy(atoms.numbers).to(torch.int32),
            positions,
            cell,
            pbc=torch.tensor([True, True, True]),
        )
        register_autograd_neighbors(system, neighbors, check_consistency=True)

        return neighbors.values.sum()

    options = NeighborListOptions(cutoff=2.0, full_list=False)
    torch.autograd.gradcheck(
        compute,
        (positions, cell, options),
        fast_mode=True,
    )

    options = NeighborListOptions(cutoff=2.0, full_list=True)
    torch.autograd.gradcheck(
        compute,
        (positions, cell, options),
        fast_mode=True,
    )


@pytest.mark.skipif(not HAVE_ASE, reason="this tests requires ASE neighbor list")
def test_neighbor_autograd_errors():
    n_atoms = 20
    cell_size = 6.0
    positions = cell_size * torch.rand(
        n_atoms, 3, dtype=torch.float64, requires_grad=True
    )
    cell = cell_size * torch.eye(3, dtype=torch.float64, requires_grad=True)

    atoms = ase.Atoms(
        "C" * positions.shape[0],
        positions=positions.detach().numpy(),
        cell=cell.detach().numpy(),
        pbc=True,
    )
    options = NeighborListOptions(cutoff=2.0, full_list=False)
    neighbors = _compute_ase_neighbors(
        atoms, options, dtype=torch.float64, device="cpu"
    )
    system = System(
        torch.from_numpy(atoms.numbers).to(torch.int32),
        positions,
        cell,
        pbc=torch.tensor([True, True, True]),
    )
    register_autograd_neighbors(system, neighbors)

    message = (
        "`neighbors` is already part of a computational graph, "
        "detach it before calling `register_autograd_neighbors\\(\\)`"
    )
    with pytest.raises(ValueError, match=message):
        register_autograd_neighbors(system, neighbors)

    message = (
        r"one neighbor pair does not match its metadata: the pair between atom \d+ and "
        r"atom \d+ for the \[.*?\] cell shift should have a distance vector "
        r"of \[.*?\] but has a distance vector of \[.*?\]"
    )
    neighbors = _compute_ase_neighbors(
        atoms, options, dtype=torch.float64, device="cpu"
    )
    neighbors.values[:] *= 3
    with pytest.raises(ValueError, match=message):
        register_autograd_neighbors(system, neighbors, check_consistency=True)

    neighbors = _compute_ase_neighbors(
        atoms, options, dtype=torch.float64, device="cpu"
    )
    message = (
        "`system` and `neighbors` must have the same dtype, "
        "got torch.float32 and torch.float64"
    )
    system = System(
        torch.from_numpy(atoms.numbers).to(torch.int32),
        positions.to(torch.float32),
        cell.to(torch.float32),
        pbc=torch.tensor([True, True, True]),
    )
    with pytest.raises(ValueError, match=message):
        register_autograd_neighbors(system, neighbors, check_consistency=True)

    message = "`system` and `neighbors` must be on the same device, got meta and cpu"
    system = System(
        torch.from_numpy(atoms.numbers).to(torch.int32).to(torch.device("meta")),
        positions.to(torch.device("meta")),
        cell.to(torch.device("meta")),
        pbc=torch.tensor([True, True, True]).to(torch.device("meta")),
    )
    with pytest.raises(ValueError, match=message):
        register_autograd_neighbors(system, neighbors, check_consistency=True)
