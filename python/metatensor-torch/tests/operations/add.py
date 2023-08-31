import torch

import metatensor.torch

from .data import load_data


def check_operation(add):
    tensor = load_data("qm7-power-spectrum.npz")
    sum_tensor = add(tensor, tensor)
    assert metatensor.torch.equal_metadata(sum_tensor, tensor)
    assert metatensor.torch.allclose(sum_tensor, metatensor.torch.multiply(tensor, 2))


def test_operation_as_python():
    check_operation(metatensor.torch.add)


def test_operation_as_torch_script():
    check_operation(torch.jit.script(metatensor.torch.add))
