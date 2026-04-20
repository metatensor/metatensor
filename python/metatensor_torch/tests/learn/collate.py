import torch

import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.learn.data import DataLoader, Dataset, group_and_join


def test_group_and_join_torch_tensormaps():
    """
    Tests that data of arbitrary types is collated correctly using
    `group_and_join` collate fxn. Specifically checks that TensorMaps imported
    from metatensor-torch are handled correctly.
    """

    dset = Dataset(
        sample_indices=list(range(3)),
        a=["a" * i for i in range(3)],
        x=[torch.ones(1, 2) * i for i in range(3)],
        y=[
            TensorMap(
                keys=Labels(names=["a"], values=torch.tensor([0]).reshape(-1, 1)),
                blocks=[
                    TensorBlock(
                        values=torch.ones((1, 1)),
                        samples=Labels(["sample_index"], torch.tensor([[0]])),
                        components=[],
                        properties=Labels(["p"], torch.tensor([[0]])),
                    )
                ],
            ),
            TensorMap(
                keys=Labels(names=["a"], values=torch.tensor([0]).reshape(-1, 1)),
                blocks=[
                    TensorBlock(
                        values=torch.ones((1, 1)),
                        samples=Labels(["sample_index"], torch.tensor([[1]])),
                        components=[],
                        properties=Labels(["p"], torch.tensor([[0]])),
                    )
                ],
            ),
            TensorMap(
                keys=Labels(names=["a"], values=torch.tensor([0]).reshape(-1, 1)),
                blocks=[
                    TensorBlock(
                        values=torch.ones((1, 1)),
                        samples=Labels(["sample_index"], torch.tensor([[2]])),
                        components=[],
                        properties=Labels(["p"], torch.tensor([[0]])),
                    )
                ],
            ),
        ],
    )
    batch_idxs = [0, 2]  # i.e. batch size 2
    batch = group_and_join([dset[i] for i in batch_idxs])

    assert batch.sample_indices == (0, 2)
    assert batch.a == ("", "aa")
    assert torch.all(batch.x == torch.tensor([[0, 0], [2, 2]]))

    # Check TensorMap joined correctly
    target_tensor = TensorMap(
        keys=Labels(names=["a"], values=torch.tensor([0]).reshape(-1, 1)),
        blocks=[
            TensorBlock(
                values=torch.ones((2, 1)),
                samples=Labels(["sample_index"], torch.tensor([[0], [2]])),
                components=[],
                properties=Labels(["p"], torch.tensor([[0]])),
            )
        ],
    )
    assert mts.equal(batch.y, target_tensor)


def test_group_and_join_script_object():
    """
    Tests that data of arbitrary types is collated correctly using
    `group_and_join` collate fxn. Specifically, this tests that a torch script
    object that isn't a TensorMap is handled correctly.
    """
    # Create a metatensor.torch.TensorBlock - i.e. a torchscript object that
    # isn't a TensorMap
    tensorblock = TensorBlock(
        values=torch.ones((1, 1)),
        samples=Labels(names=["sample_index"], values=torch.tensor([[0]])),
        components=[],
        properties=Labels(names=["p"], values=torch.tensor([[0]])),
    )
    tensormap = TensorMap(
        keys=Labels(names=["a"], values=torch.tensor([0]).reshape(-1, 1)),
        blocks=[tensorblock.copy()],
    )

    dset = Dataset(a=[tensorblock], b=[tensormap])
    dloader = DataLoader(dset, batch_size=2, collate_fn=group_and_join)
    next(iter(dloader))  # this errors if type-check logic not correct
