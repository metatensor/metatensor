import torch
from packaging import version

import equistore.torch


def check_operation(block_from_array):
    values = torch.arange(42).reshape(2, 3, 7).to(torch.float64)
    block = block_from_array(values)

    # check output type
    assert isinstance(block, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert block._type().name() == "TensorBlock"

    # check values
    assert torch.equal(block.values, values)


def test_operation_as_python():
    check_operation(equistore.torch.block_from_array)


def test_operation_as_torch_script():
    check_operation(torch.jit.script(equistore.torch.block_from_array))
