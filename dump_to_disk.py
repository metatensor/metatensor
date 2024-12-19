import os

import torch

from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import NeighborListOptions, System, save, load
import tqdm

for i in tqdm.tqdm(range(10000)):
    system = System(
        types=torch.tensor([1, 2, 3, 4]),
        positions=torch.rand((4, 3), dtype=torch.float64),
        cell=torch.rand((3, 3), dtype=torch.float64),
        pbc=torch.tensor([True, True, True], dtype=torch.bool),
    )
    nl_block = TensorBlock(
        values=torch.rand(2000, 3, 1, dtype=torch.float64),
        samples=Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            torch.arange(2000 * 5, dtype=torch.int64).reshape(2000, 5),
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
    system.add_neighbor_list(
        NeighborListOptions(cutoff=3.5, full_list=True, strict=True),
        nl_block,
    )

    save(os.path.join("dump", f"system_{i}.npz"), system)


for i in tqdm.tqdm(range(10000)):
    load(os.path.join("dump", f"system_{i}.npz"))
