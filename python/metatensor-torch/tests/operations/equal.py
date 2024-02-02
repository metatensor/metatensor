import io

import torch

import metatensor.torch

from .data import load_data


def check_operation(equal):
    tensor = load_data("qm7-power-spectrum.npz")
    assert equal(tensor, tensor)


def check_operation_raise(equal_raise):
    tensor = load_data("qm7-power-spectrum.npz")
    equal_raise(tensor, tensor)


def check_operation_block(equal_block):
    tensor = load_data("qm7-power-spectrum.npz")
    assert equal_block(tensor.block(0), tensor.block(0))


def check_operation_block_raise(equal_block_raise):
    tensor = load_data("qm7-power-spectrum.npz")
    equal_block_raise(tensor.block(0), tensor.block(0))


def test_operations_as_python():
    check_operation(metatensor.torch.equal)
    check_operation_raise(metatensor.torch.equal_raise)
    check_operation_block(metatensor.torch.equal_block)
    check_operation_block_raise(metatensor.torch.equal_block_raise)


def test_operations_as_torch_script():
    print(type(metatensor.torch.equal))
    scripted = torch.jit.script(metatensor.torch.equal)
    check_operation(scripted)
    scripted = torch.jit.script(metatensor.torch.equal_raise)
    check_operation_raise(scripted)
    scripted = torch.jit.script(metatensor.torch.equal_block)
    check_operation_block(scripted)
    scripted = torch.jit.script(metatensor.torch.equal_block_raise)
    check_operation_block_raise(scripted)


def test_save_load():
    scripted = torch.jit.script(metatensor.torch.equal)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
    scripted = torch.jit.script(metatensor.torch.equal_raise)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
    scripted = torch.jit.script(metatensor.torch.equal_block)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
    scripted = torch.jit.script(metatensor.torch.equal_block_raise)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
