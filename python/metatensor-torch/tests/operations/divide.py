import torch

import metatensor.torch

from .data import load_data


def check_operation(divide):
    tensor = load_data("qm7-power-spectrum.npz")
    quotient_tensor = divide(tensor, tensor)
    assert metatensor.torch.equal_metadata(quotient_tensor, tensor)
    assert torch.allclose(
        torch.nan_to_num(quotient_tensor.block(0).values, 1.0),  # replace nan with 1.0
        torch.ones_like(quotient_tensor.block(0).values),
    )


def test_operation_as_python():
    check_operation(metatensor.torch.divide)


def test_operation_as_torch_script():
    check_operation(torch.jit.script(metatensor.torch.divide))
