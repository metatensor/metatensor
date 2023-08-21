import torch

import equistore.torch

from .data import load_data


def check_operation(equal_metadata):
    tensor = load_data("qm7-power-spectrum.npz")
    assert equal_metadata(tensor, tensor)


def check_operation_raise(equal_metadata_raise):
    tensor = load_data("qm7-power-spectrum.npz")
    equal_metadata_raise(tensor, tensor)


def check_operation_block(equal_metadata_block):
    tensor = load_data("qm7-power-spectrum.npz")
    assert equal_metadata_block(tensor.block(0), tensor.block(0))


def check_operation_block_raise(equal_metadata_block_raise):
    tensor = load_data("qm7-power-spectrum.npz")
    equal_metadata_block_raise(tensor.block(0), tensor.block(0))


def test_operations_as_python():
    check_operation(equistore.torch.equal_metadata)
    check_operation_raise(equistore.torch.equal_metadata_raise)
    check_operation_block(equistore.torch.equal_metadata_block)
    check_operation_block_raise(equistore.torch.equal_metadata_block_raise)


def test_operations_as_torch_script():
    scripted = torch.jit.script(equistore.torch.equal_metadata)
    check_operation(scripted)
    scripted = torch.jit.script(equistore.torch.equal_metadata_raise)
    check_operation_raise(scripted)
    scripted = torch.jit.script(equistore.torch.equal_metadata_block)
    check_operation_block(scripted)
    scripted = torch.jit.script(equistore.torch.equal_metadata_block_raise)
    check_operation_block_raise(scripted)
