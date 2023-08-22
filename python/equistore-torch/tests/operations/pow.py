import torch

import equistore.torch

from .data import load_data


def test_operation_as_python():
    tensor = load_data("qm7-power-spectrum.npz")
    power_tensor = equistore.torch.pow(tensor, 2.0)
    assert equistore.torch.equal_metadata(power_tensor, tensor)
    assert equistore.torch.allclose(
        power_tensor, equistore.torch.multiply(tensor, tensor)
    )


def test_operation_as_torch_script():
    tensor = load_data("qm7-power-spectrum.npz")
    power_tensor = torch.jit.script(equistore.torch.pow)(tensor, 2.0)
    assert equistore.torch.equal_metadata(power_tensor, tensor)
    assert equistore.torch.allclose(
        power_tensor, equistore.torch.multiply(tensor, tensor)
    )
