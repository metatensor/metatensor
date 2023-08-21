import torch

import equistore.torch

from .data import load_data


def check_operation(equal_metadata):
    tensor = load_data("qm7-power-spectrum.npz")
    assert equal_metadata(tensor, tensor)


def test_operation_as_python():
    check_operation(equistore.torch.equal_metadata)


def test_operation_as_torch_script():
    scripted = torch.jit.script(equistore.torch.equal_metadata)

    check_operation(scripted)
