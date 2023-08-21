import torch

import equistore.torch

from .data import load_data


def check_operation(allclose):
    tensor = load_data("qm7-power-spectrum.npz")
    assert allclose(tensor, tensor)


def check_operation_raise(allclose_raise):
    tensor = load_data("qm7-power-spectrum.npz")
    allclose_raise(tensor, tensor)


def check_operation_block(allclose_block):
    tensor = load_data("qm7-power-spectrum.npz")
    assert allclose_block(tensor.block(0), tensor.block(0))


def check_operation_block_raise(allclose_block_raise):
    tensor = load_data("qm7-power-spectrum.npz")
    allclose_block_raise(tensor.block(0), tensor.block(0))


def test_operations_as_python():
    check_operation(equistore.torch.allclose)
    check_operation_raise(equistore.torch.allclose_raise)
    check_operation_block(equistore.torch.allclose_block)
    check_operation_block_raise(equistore.torch.allclose_block_raise)


def test_operations_as_torch_script():
    scripted = torch.jit.script(equistore.torch.allclose)
    check_operation(scripted)
    scripted = torch.jit.script(equistore.torch.allclose_raise)
    check_operation_raise(scripted)
    scripted = torch.jit.script(equistore.torch.allclose_block)
    check_operation_block(scripted)
    scripted = torch.jit.script(equistore.torch.allclose_block_raise)
    check_operation_block_raise(scripted)
