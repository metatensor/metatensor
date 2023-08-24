import torch

import equistore.torch

from .data import load_data


def check_operation(multiply):
    tensor = load_data("qm7-power-spectrum.npz")
    product_tensor = multiply(tensor, tensor)
    assert equistore.torch.equal_metadata(product_tensor, tensor)
    assert torch.allclose(product_tensor.block(0).values, tensor.block(0).values ** 2)


def test_operation_as_python():
    check_operation(equistore.torch.multiply)


def test_operation_as_torch_script():
    check_operation(torch.jit.script(equistore.torch.multiply))
