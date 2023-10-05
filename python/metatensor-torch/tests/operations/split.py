import torch
from packaging import version

import metatensor.torch
from metatensor.torch import Labels, TensorMap


def check_operation(split):
    tensor = TensorMap(
        keys=Labels.single(),
        blocks=[
            metatensor.torch.block_from_array(torch.tensor([[0, 1, 2], [3, 4, 5]]))
        ],
    )
    split_labels_samples = [
        Labels(names=["sample"], values=torch.tensor([[0]])),
        Labels(names=["sample"], values=torch.tensor([[1]])),
    ]
    split_labels_properties = [
        Labels(names=["property"], values=torch.tensor([[0], [1]])),
        Labels(names=["property"], values=torch.tensor([[2]])),
    ]
    split_tensors_samples = split(
        tensor,
        axis="samples",
        grouped_labels=split_labels_samples,
    )
    split_tensors_properties = split(
        tensor,
        axis="properties",
        grouped_labels=split_labels_properties,
    )

    # check type
    assert isinstance(split_tensors_samples, list)
    assert isinstance(split_tensors_samples[0], torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert split_tensors_samples[0]._type().name() == "TensorMap"

    assert isinstance(split_tensors_properties, list)
    assert isinstance(split_tensors_properties[0], torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
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


def check_operation_block(split_block):
    block = metatensor.torch.block_from_array(torch.tensor([[0, 1, 2], [3, 4, 5]]))
    split_labels_samples = [
        Labels(names=["sample"], values=torch.tensor([[0]])),
        Labels(names=["sample"], values=torch.tensor([[1]])),
    ]
    split_labels_properties = [
        Labels(names=["property"], values=torch.tensor([[0], [1]])),
        Labels(names=["property"], values=torch.tensor([[2]])),
    ]
    split_blocks_samples = split_block(
        block,
        axis="samples",
        grouped_labels=split_labels_samples,
    )
    split_blocks_properties = split_block(
        block,
        axis="properties",
        grouped_labels=split_labels_properties,
    )

    # check type
    assert isinstance(split_blocks_samples, list)
    assert isinstance(split_blocks_samples[0], torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert split_blocks_samples[0]._type().name() == "TensorBlock"

    assert isinstance(split_blocks_properties, list)
    assert isinstance(split_blocks_properties[0], torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert split_blocks_properties[0]._type().name() == "TensorBlock"

    # check values
    assert torch.equal(split_blocks_samples[0].values, torch.tensor([[0, 1, 2]]))
    assert torch.equal(split_blocks_samples[1].values, torch.tensor([[3, 4, 5]]))
    assert torch.equal(
        split_blocks_properties[0].values, torch.tensor([[0, 1], [3, 4]])
    )
    assert torch.equal(split_blocks_properties[1].values, torch.tensor([[2], [5]]))


def test_operations_as_python():
    check_operation(metatensor.torch.split)
    check_operation_block(metatensor.torch.split_block)


def test_operations_as_torch_script():
    check_operation(torch.jit.script(metatensor.torch.split))
    check_operation_block(torch.jit.script(metatensor.torch.split_block))
