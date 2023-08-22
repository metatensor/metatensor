import torch

import equistore.torch

from .data import load_data


def test_operation_as_python():
    tensor = load_data("qm7-power-spectrum.npz")
    product_tensor = equistore.torch.multiply(tensor, tensor)
    assert equistore.torch.equal_metadata(product_tensor, tensor)


def test_operation_as_torch_script():
    tensor = load_data("qm7-power-spectrum.npz")
    product_tensor = torch.jit.script(equistore.torch.multiply)(tensor, tensor)
    assert equistore.torch.equal_metadata(product_tensor, tensor)
