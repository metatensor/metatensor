import torch

import equistore.torch

from .data import load_data


def check_operation(pow):
    tensor = load_data("qm7-power-spectrum.npz")
    power_tensor = pow(tensor, 2.0)
    assert equistore.torch.equal_metadata(power_tensor, tensor)
    assert equistore.torch.allclose(
        power_tensor, equistore.torch.multiply(tensor, tensor)
    )


def test_operation_as_python():
    check_operation(equistore.torch.pow)


def test_operation_as_torch_script():
    check_operation(torch.jit.script(equistore.torch.pow))
