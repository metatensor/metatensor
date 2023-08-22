import torch

import equistore.torch

from .data import load_data


def check_operation(subtract):
    tensor = load_data("qm7-power-spectrum.npz")
    difference_tensor = subtract(tensor, tensor)
    assert equistore.torch.equal_metadata(difference_tensor, tensor)
    assert torch.allclose(
        difference_tensor.block(0).values,
        torch.zeros_like(difference_tensor.block(0).values),
    )


def test_operation_as_python():
    check_operation(equistore.torch.subtract)


def test_operation_as_torch_script():
    check_operation(torch.jit.script(equistore.torch.subtract))
