import io
import os

import pytest
import torch

import metatensor.torch as mts
from metatensor.torch import Labels, TensorMap


def test_slice():
    tensor = TensorMap(
        keys=Labels.single(),
        blocks=[mts.block_from_array(torch.tensor([[0, 1, 2], [3, 4, 5]]))],
    )
    samples = Labels(names=["sample"], values=torch.tensor([[1]]))
    properties = Labels(names=["property"], values=torch.tensor([[1]]))
    sliced_tensor_samples = mts.slice(tensor, axis="samples", selection=samples)
    sliced_tensor_properties = mts.slice(
        tensor, axis="properties", selection=properties
    )

    # check type
    assert isinstance(sliced_tensor_samples, torch.ScriptObject)
    assert sliced_tensor_samples._type().name() == "TensorMap"
    assert isinstance(sliced_tensor_properties, torch.ScriptObject)
    assert sliced_tensor_properties._type().name() == "TensorMap"

    # check values
    assert torch.equal(sliced_tensor_samples.block().values, torch.tensor([[3, 4, 5]]))
    assert torch.equal(
        sliced_tensor_properties.block().values, torch.tensor([[1], [4]])
    )


def test_slice_block():
    block = mts.block_from_array(torch.tensor([[0, 1, 2], [3, 4, 5]]))
    samples = Labels(names=["sample"], values=torch.tensor([[1]]))
    properties = Labels(names=["property"], values=torch.tensor([[1]]))
    sliced_block_samples = mts.slice_block(block, axis="samples", selection=samples)
    sliced_block_properties = mts.slice_block(
        block, axis="properties", selection=properties
    )

    # check type
    assert isinstance(sliced_block_samples, torch.ScriptObject)
    assert sliced_block_samples._type().name() == "TensorBlock"
    assert isinstance(sliced_block_properties, torch.ScriptObject)
    assert sliced_block_properties._type().name() == "TensorBlock"

    # check values
    assert torch.equal(sliced_block_samples.values, torch.tensor([[3, 4, 5]]))
    assert torch.equal(sliced_block_properties.values, torch.tensor([[1], [4]]))


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.slice, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(mts.slice_block, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
