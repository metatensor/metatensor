"""
Module for testing the custom collate functions in :py:module:`collate`.
"""
import numpy as np
import pytest


torch = pytest.importorskip("torch")

import metatensor  # noqa: E402
from metatensor import Labels, TensorBlock, TensorMap  # noqa: E402
from metatensor.learn.data import Dataset, group, group_and_join  # noqa: E402


def test_group_object_types():
    """
    Tests that data of arbitrary types is collated correctly using `group`
    collate fxn.
    """
    dset = Dataset(
        a=["a" * i for i in range(10)],
        c=[(3, 4, i) for i in range(10)],
        sample_indices=list(range(10)),
        b=[0 for _ in range(10)],
        x=[torch.ones(1, 1) * i for i in range(10)],
    )
    batch_idxs = [0, 4, 5]  # i.e. batch size 3
    batch = group([dset[i] for i in batch_idxs])

    assert batch.sample_indices == (0, 4, 5)
    assert batch.a == ("", "aaaa", "aaaaa")
    assert batch.c == ((3, 4, 0), (3, 4, 4), (3, 4, 5))
    assert batch.b == (0, 0, 0)
    assert all([batch.x[i] == torch.tensor([[s]]) for i, s in enumerate(batch_idxs)])


def test_group_and_join_object_types():
    """
    Tests that data of arbitrary types is collated correctly using
    `group_and_join` collate fxn.
    """
    dset = Dataset(
        a=["a" * i for i in range(10)],
        c=[(3, 4, i) for i in range(10)],
        sample_indices=list(range(10)),
        b=[0 for _ in range(10)],
        x=[torch.ones(1, 2) * i for i in range(10)],
    )
    batch_idxs = [0, 4, 5]  # i.e. batch size 3
    batch = group_and_join([dset[i] for i in batch_idxs])

    assert batch.sample_indices == (0, 4, 5)
    assert batch.a == ("", "aaaa", "aaaaa")
    assert batch.c == ((3, 4, 0), (3, 4, 4), (3, 4, 5))
    assert batch.b == (0, 0, 0)
    assert torch.all(batch.x == torch.tensor([[0, 0], [4, 4], [5, 5]]))


def test_group_and_join_tensormaps():
    """
    Tests that data of arbitrary types is collated correctly using
    `group_and_join` collate fxn.
    """
    dset = Dataset(
        sample_indices=list(range(3)),
        a=["a" * i for i in range(3)],
        x=[torch.ones(1, 2) * i for i in range(3)],
        y=[
            TensorMap(
                keys=Labels(names=["a"], values=np.array([0]).reshape(-1, 1)),
                blocks=[
                    TensorBlock(
                        values=np.ones((1, 1)),
                        samples=Labels(["sample_index"], np.array([[0]])),
                        components=[],
                        properties=Labels(["p"], np.array([[0]])),
                    )
                ],
            ),
            TensorMap(
                keys=Labels(names=["a"], values=np.array([0]).reshape(-1, 1)),
                blocks=[
                    TensorBlock(
                        values=np.ones((1, 1)),
                        samples=Labels(["sample_index"], np.array([[1]])),
                        components=[],
                        properties=Labels(["p"], np.array([[0]])),
                    )
                ],
            ),
            TensorMap(
                keys=Labels(names=["a"], values=np.array([0]).reshape(-1, 1)),
                blocks=[
                    TensorBlock(
                        values=np.ones((1, 1)),
                        samples=Labels(["sample_index"], np.array([[2]])),
                        components=[],
                        properties=Labels(["p"], np.array([[0]])),
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
        keys=Labels(names=["a"], values=np.array([0]).reshape(-1, 1)),
        blocks=[
            TensorBlock(
                values=np.ones((2, 1)),
                samples=Labels(["sample_index", "tensor"], np.array([[0, 0], [2, 1]])),
                components=[],
                properties=Labels(["p"], np.array([[0]])),
            )
        ],
    )
    assert metatensor.equal(batch.y, target_tensor)


def test_group_and_join_torch_tensormaps():
    """
    Tests that data of arbitrary types is collated correctly using
    `group_and_join` collate fxn. Specifically checks that TensorMaps imported
    from metatensor-torch are handled correctly.
    """
    try:
        from metatensor.torch import Labels, TensorBlock, TensorMap
    except ImportError:
        pytest.skip("metatensor-torch not installed")
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
                samples=Labels(
                    ["sample_index", "tensor"], torch.tensor([[0, 0], [2, 1]])
                ),
                components=[],
                properties=Labels(["p"], torch.tensor([[0]])),
            )
        ],
    )
    assert metatensor.equal(batch.y, target_tensor)


def test_group_and_join_tensormaps_different_keys_error():
    """
    Tests that attempting to collate TensorMaps by `group_and_join` raises an
    error using the join default kwargs when the keys are different.
    """
    dset = Dataset(
        sample_indices=list(range(3)),
        y=[
            TensorMap(
                keys=Labels(names=["a"], values=np.array([0]).reshape(-1, 1)),
                blocks=[
                    TensorBlock(
                        values=np.ones((1, 1)),
                        samples=Labels(["sample_index"], np.array([[0]])),
                        components=[],
                        properties=Labels(["p"], np.array([[0]])),
                    )
                ],
            ),
            TensorMap(
                keys=Labels(names=["a"], values=np.array([1]).reshape(-1, 1)),
                blocks=[
                    TensorBlock(
                        values=np.ones((1, 1)),
                        samples=Labels(["sample_index"], np.array([[1]])),
                        components=[],
                        properties=Labels(["p"], np.array([[0]])),
                    )
                ],
            ),
            TensorMap(
                keys=Labels(names=["a"], values=np.array([2]).reshape(-1, 1)),
                blocks=[
                    TensorBlock(
                        values=np.ones((1, 1)),
                        samples=Labels(["sample_index"], np.array([[2]])),
                        components=[],
                        properties=Labels(["p"], np.array([[0]])),
                    )
                ],
            ),
        ],
    )
    batch_idxs = [0, 2]  # i.e. batch size 2
    message = "inputs to 'join' should have the same keys"
    with pytest.raises(metatensor.NotEqualError, match=message):
        group_and_join([dset[i] for i in batch_idxs])


def test_group_and_join_tensormaps_different_keys_union():
    """
    Tests that attempting to collate TensorMaps by `group_and_join` raises an
    error using the join default kwargs when the keys are different.
    """
    dset = Dataset(
        sample_indices=list(range(3)),
        y=[
            TensorMap(
                keys=Labels(names=["a"], values=np.array([0]).reshape(-1, 1)),
                blocks=[
                    TensorBlock(
                        values=np.ones((1, 1)),
                        samples=Labels(["sample_index"], np.array([[0]])),
                        components=[],
                        properties=Labels(["p"], np.array([[0]])),
                    )
                ],
            ),
            TensorMap(
                keys=Labels(names=["a"], values=np.array([1]).reshape(-1, 1)),
                blocks=[
                    TensorBlock(
                        values=np.ones((1, 1)),
                        samples=Labels(["sample_index"], np.array([[1]])),
                        components=[],
                        properties=Labels(["p"], np.array([[0]])),
                    )
                ],
            ),
            TensorMap(
                keys=Labels(names=["a"], values=np.array([2]).reshape(-1, 1)),
                blocks=[
                    TensorBlock(
                        values=np.ones((1, 1)),
                        samples=Labels(["sample_index"], np.array([[2]])),
                        components=[],
                        properties=Labels(["p"], np.array([[0]])),
                    )
                ],
            ),
        ],
    )
    batch_idxs = [0, 2]  # i.e. batch size 2
    batch = group_and_join(
        [dset[i] for i in batch_idxs],
        join_kwargs={"different_keys": "union", "remove_tensor_name": True},
    )
    assert batch.sample_indices == (0, 2)

    # Check TensorMap joined correctly
    target_tensor = TensorMap(
        keys=Labels(names=["a"], values=np.array([0, 2]).reshape(-1, 1)),
        blocks=[
            TensorBlock(
                values=np.ones((1, 1)),
                samples=Labels(["sample_index"], np.array([[0]])),
                components=[],
                properties=Labels(["p"], np.array([[0]])),
            ),
            TensorBlock(
                values=np.ones((1, 1)),
                samples=Labels(["sample_index"], np.array([[2]])),
                components=[],
                properties=Labels(["p"], np.array([[0]])),
            ),
        ],
    )
    assert metatensor.equal(batch.y, target_tensor)
