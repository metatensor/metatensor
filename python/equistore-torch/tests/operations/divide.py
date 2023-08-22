import torch

import equistore.torch

from .data import load_data


def test_operation_as_python():
    tensor = load_data("qm7-power-spectrum.npz")
    quotient_tensor = equistore.torch.divide(tensor, tensor)
    assert equistore.torch.equal_metadata(quotient_tensor, tensor)


def test_operation_as_torch_script():
    tensor = load_data("qm7-power-spectrum.npz")
    quotient_tensor = torch.jit.script(equistore.torch.divide)(tensor, tensor)
    assert equistore.torch.equal_metadata(quotient_tensor, tensor)
