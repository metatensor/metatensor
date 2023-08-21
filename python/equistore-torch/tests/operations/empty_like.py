import torch

import equistore.torch

from .data import load_data


def test_operation_as_python():
    tensor = load_data("qm7-power-spectrum.npz")
    zero_tensor = equistore.torch.empty_like(tensor)
    assert equistore.torch.equal_metadata(zero_tensor, tensor)


def test_operation_as_torch_script():
    tensor = load_data("qm7-power-spectrum.npz")
    zero_tensor = torch.jit.script(equistore.torch.empty_like)(tensor)
    assert equistore.torch.equal_metadata(zero_tensor, tensor)
