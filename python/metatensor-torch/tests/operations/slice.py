import io

import torch
from packaging import version

import metatensor.torch
from metatensor.torch import Labels, TensorMap


def check_operation(slice):
    tensor = TensorMap(
        keys=Labels.single(),
        blocks=[
            metatensor.torch.block_from_array(torch.tensor([[0, 1, 2], [3, 4, 5]]))
        ],
    )
    samples = Labels(
        names=["sample"],
        values=torch.tensor([[1]]),
    )
    properties = Labels(
        names=["property"],
        values=torch.tensor([[1]]),
    )
    sliced_tensor_samples = slice(
        tensor,
        axis="samples",
        labels=samples,
    )
    sliced_tensor_properties = slice(
        tensor,
        axis="properties",
        labels=properties,
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


def check_operation_block(slice_block):
    block = metatensor.torch.block_from_array(torch.tensor([[0, 1, 2], [3, 4, 5]]))
    samples = Labels(
        names=["sample"],
        values=torch.tensor([[1]]),
    )
    properties = Labels(
        names=["property"],
        values=torch.tensor([[1]]),
    )
    sliced_block_samples = slice_block(
        block,
        axis="samples",
        labels=samples,
    )
    sliced_block_properties = slice_block(
        block,
        axis="properties",
        labels=properties,
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


def test_operations_as_python():
    check_operation(metatensor.torch.slice)
    check_operation_block(metatensor.torch.slice_block)


def test_operations_as_torch_script():
    check_operation(torch.jit.script(metatensor.torch.slice))
    check_operation_block(torch.jit.script(metatensor.torch.slice_block))


def test_save_load():
    scripted = torch.jit.script(metatensor.torch.slice)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
    scripted = torch.jit.script(metatensor.torch.slice_block)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
