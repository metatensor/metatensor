import torch

import equistore.torch

from .data import load_data


def test_operation_as_python():
    tensor = load_data("qm7-power-spectrum.npz")
    sum_tensor = equistore.torch.add(tensor, tensor)
    assert equistore.torch.equal_metadata(sum_tensor, tensor)
    assert equistore.torch.allclose(sum_tensor, equistore.torch.multiply(tensor, 2))


def test_operation_as_torch_script():
    tensor = load_data("qm7-power-spectrum.npz")
    sum_tensor = torch.jit.script(equistore.torch.add)(tensor, tensor)
    assert equistore.torch.equal_metadata(sum_tensor, tensor)
    assert equistore.torch.allclose(sum_tensor, equistore.torch.multiply(tensor, 2))
