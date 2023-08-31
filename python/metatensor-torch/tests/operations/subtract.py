import torch

import metatensor.torch

from .data import load_data


def check_operation(subtract):
    tensor = load_data("qm7-power-spectrum.npz")
    difference_tensor = subtract(tensor, tensor)
    assert metatensor.torch.equal_metadata(difference_tensor, tensor)
    assert torch.allclose(
        difference_tensor.block(0).values,
        torch.zeros_like(difference_tensor.block(0).values),
    )


def test_operation_as_python():
    check_operation(metatensor.torch.subtract)


def test_operation_as_torch_script():
    check_operation(torch.jit.script(metatensor.torch.subtract))
