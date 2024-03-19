import io

import torch
from packaging import version

import metatensor.torch
from metatensor.torch import Labels, TensorMap


def test_slice():
    tensor = TensorMap(
        keys=Labels.single(),
        blocks=[
            metatensor.torch.block_from_array(torch.tensor([[0, 1, 2], [3, 4, 5]]))
        ],
    )
    samples = Labels(names=["sample"], values=torch.tensor([[1]]))
    properties = Labels(names=["property"], values=torch.tensor([[1]]))
    sliced_tensor_samples = metatensor.torch.slice(
        tensor, axis="samples", labels=samples
    )
    sliced_tensor_properties = metatensor.torch.slice(
        tensor, axis="properties", labels=properties
    )

    # check type
    assert isinstance(sliced_tensor_samples, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert sliced_tensor_samples._type().name() == "TensorMap"
    assert isinstance(sliced_tensor_properties, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert sliced_tensor_properties._type().name() == "TensorMap"

    # check values
    assert torch.equal(sliced_tensor_samples.block().values, torch.tensor([[3, 4, 5]]))
    assert torch.equal(
        sliced_tensor_properties.block().values, torch.tensor([[1], [4]])
    )


def test_slice_block():
    block = metatensor.torch.block_from_array(torch.tensor([[0, 1, 2], [3, 4, 5]]))
    samples = Labels(names=["sample"], values=torch.tensor([[1]]))
    properties = Labels(names=["property"], values=torch.tensor([[1]]))
    sliced_block_samples = metatensor.torch.slice_block(
        block, axis="samples", labels=samples
    )
    sliced_block_properties = metatensor.torch.slice_block(
        block, axis="properties", labels=properties
    )

    # check type
    assert isinstance(sliced_block_samples, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert sliced_block_samples._type().name() == "TensorBlock"
    assert isinstance(sliced_block_properties, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert sliced_block_properties._type().name() == "TensorBlock"

    # check values
    assert torch.equal(sliced_block_samples.values, torch.tensor([[3, 4, 5]]))
    assert torch.equal(sliced_block_properties.values, torch.tensor([[1], [4]]))


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.slice, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.slice_block, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
