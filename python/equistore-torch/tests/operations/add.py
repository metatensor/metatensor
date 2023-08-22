import torch

import equistore.torch

from .data import load_data


def check_operation(add):
    tensor = load_data("qm7-power-spectrum.npz")
    sum_tensor = add(tensor, tensor)
    assert equistore.torch.equal_metadata(sum_tensor, tensor)
    assert equistore.torch.allclose(sum_tensor, equistore.torch.multiply(tensor, 2))


def test_operation_as_python():
    check_operation(equistore.torch.add)


def test_operation_as_torch_script():
    check_operation(torch.jit.script(equistore.torch.add))
