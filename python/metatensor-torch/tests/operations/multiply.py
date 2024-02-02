import io

import torch

import metatensor.torch

from .data import load_data


def check_operation(multiply):
    tensor = load_data("qm7-power-spectrum.npz")
    product_tensor = multiply(tensor, tensor)
    assert metatensor.torch.equal_metadata(product_tensor, tensor)
    assert torch.allclose(product_tensor.block(0).values, tensor.block(0).values ** 2)


def test_operation_as_python():
    check_operation(metatensor.torch.multiply)


def test_operation_as_torch_script():
    check_operation(torch.jit.script(metatensor.torch.multiply))


def test_save():
    scripted = torch.jit.script(metatensor.torch.multiply)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
