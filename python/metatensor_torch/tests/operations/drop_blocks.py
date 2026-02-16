import io
import os

import pytest
import torch

import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap


def test_drop_blocks():
    # this only runs basic checks functionality checks, and that the code produces
    # output with the right type
    tensor = mts.load(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "metatensor_operations",
            "tests",
            "data",
            "qm7-power-spectrum.mts",
        )
    )
    tensor = tensor.keys_to_properties("neighbor_1_type")
    tensor = tensor.keys_to_properties("neighbor_2_type")

    assert tensor.keys == Labels(
        names=["center_type"], values=torch.tensor([[1], [6], [8]])
    )

    keys_to_drop = Labels(names=["center_type"], values=torch.tensor([[1], [8]]))

    tensor = mts.drop_blocks(tensor, keys_to_drop)

    # check type
    assert isinstance(tensor, torch.ScriptObject)
    assert tensor._type().name() == "TensorMap"

    # check remaining block
    expected_keys = Labels(names=["center_type"], values=torch.tensor([[6]]))
    assert tensor.keys == expected_keys


def test_drop_empty_blocks():
    """
    Define a  TensorMap in which block_2 has a zero-length dimension.
    Assert that calling drop_empty_blocks() removes block_2 from
    the resulting TensorMap.
    """

    # three blocks, with block_2 having zero samples

    block_1 = TensorBlock(
        values=torch.full((5, 3), 2.0),
        samples=Labels.range("sample", 5),
        components=[],
        properties=Labels.range("property", 3),
    )
    block_2 = TensorBlock(
        values=torch.full((0, 4), 0.0),
        samples=Labels.range("sample", 0),
        components=[],
        properties=Labels.range("property", 4),
    )
    block_3 = TensorBlock(
        values=torch.full((2, 6), 3.0),
        samples=Labels.range("sample", 2),
        components=[],
        properties=Labels.range("property", 6),
    )

    keys = Labels(names=["id"], values=torch.tensor([[0], [1], [2]]))

    # keep backup  to verify that content is unchanged for kept blocks
    bkp_block_1_values = block_1.values.clone()
    bkp_block_3_values = block_3.values.clone()

    tensor = TensorMap(keys=keys, blocks=[block_1, block_2, block_3])

    new_tensor = mts.drop_empty_blocks(tensor, copy=False)

    # check type
    assert isinstance(new_tensor, torch.ScriptObject)
    assert new_tensor._type().name() == "TensorMap"

    # check that the new tensor has the expected number of blocks and the right keys
    assert len(new_tensor) == 2
    expected_keys = Labels(names=["id"], values=torch.tensor([[0], [2]]))
    assert new_tensor.keys == expected_keys

    # check that the content of the remaining blocks is unchanged
    torch.testing.assert_close(new_tensor[0].values, bkp_block_1_values)
    torch.testing.assert_close(new_tensor[1].values, bkp_block_3_values)


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.drop_blocks, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load_drop_empty_blocks():
    """
    Test that drop_empty_blocks can be saved and loaded using TorchScript."""
    with io.BytesIO() as buffer:
        torch.jit.save(mts.drop_empty_blocks, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
