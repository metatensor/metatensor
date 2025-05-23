import io
import os

import pytest
import torch

import metatensor.torch as mts
from metatensor.torch import Labels, TensorMap


@pytest.mark.parametrize(
    "selections",
    [
        # List[ints]
        ([[0], [1]], [[0, 1], [2]]),
        # Labels
        (
            [
                Labels(names=["sample"], values=torch.tensor([[0]])),
                Labels(names=["sample"], values=torch.tensor([[1]])),
            ],
            [
                Labels(names=["property"], values=torch.tensor([[0], [1]])),
                Labels(names=["property"], values=torch.tensor([[2]])),
            ],
        ),
        (
            [  # mixed selection
                Labels(names=["sample"], values=torch.tensor([[0]])),
                [1],
            ],
            [
                torch.tensor([0, 1]),
                Labels(names=["property"], values=torch.tensor([[2]])),
            ],
        ),
    ],
)
def test_split(selections):
    tensor = TensorMap(
        keys=Labels.single(),
        blocks=[mts.block_from_array(torch.tensor([[0, 1, 2], [3, 4, 5]]))],
    )
    samples_selections, properties_selections = selections

    split_tensors_samples = mts.split(
        tensor,
        axis="samples",
        selections=samples_selections,
    )
    split_tensors_properties = mts.split(
        tensor,
        axis="properties",
        selections=properties_selections,
    )

    # check type
    assert isinstance(split_tensors_samples, list)
    assert isinstance(split_tensors_samples[0], torch.ScriptObject)
    assert split_tensors_samples[0]._type().name() == "TensorMap"

    assert isinstance(split_tensors_properties, list)
    assert isinstance(split_tensors_properties[0], torch.ScriptObject)
    assert split_tensors_properties[0]._type().name() == "TensorMap"

    # check values
    assert torch.equal(
        split_tensors_samples[0].block().values, torch.tensor([[0, 1, 2]])
    )
    assert torch.equal(
        split_tensors_samples[1].block().values, torch.tensor([[3, 4, 5]])
    )
    assert torch.equal(
        split_tensors_properties[0].block().values, torch.tensor([[0, 1], [3, 4]])
    )
    assert torch.equal(
        split_tensors_properties[1].block().values, torch.tensor([[2], [5]])
    )


def test_split_block():
    block = mts.block_from_array(torch.tensor([[0, 1, 2], [3, 4, 5]]))
    split_labels_samples = [
        Labels(names=["sample"], values=torch.tensor([[0]])),
        Labels(names=["sample"], values=torch.tensor([[1]])),
    ]
    split_labels_properties = [
        Labels(names=["property"], values=torch.tensor([[0], [1]])),
        Labels(names=["property"], values=torch.tensor([[2]])),
    ]
    split_blocks_samples = mts.split_block(
        block,
        axis="samples",
        selections=split_labels_samples,
    )
    split_blocks_properties = mts.split_block(
        block,
        axis="properties",
        selections=split_labels_properties,
    )

    # check type
    assert isinstance(split_blocks_samples, list)
    assert isinstance(split_blocks_samples[0], torch.ScriptObject)
    assert split_blocks_samples[0]._type().name() == "TensorBlock"

    assert isinstance(split_blocks_properties, list)
    assert isinstance(split_blocks_properties[0], torch.ScriptObject)
    assert split_blocks_properties[0]._type().name() == "TensorBlock"

    # check values
    assert torch.equal(split_blocks_samples[0].values, torch.tensor([[0, 1, 2]]))
    assert torch.equal(split_blocks_samples[1].values, torch.tensor([[3, 4, 5]]))
    assert torch.equal(
        split_blocks_properties[0].values, torch.tensor([[0, 1], [3, 4]])
    )
    assert torch.equal(split_blocks_properties[1].values, torch.tensor([[2], [5]]))


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.split, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(mts.split_block, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
