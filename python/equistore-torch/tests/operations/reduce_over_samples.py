import torch
from packaging import version

import equistore.torch

from .data import load_data


def check_operation(reduce_over_samples):
    tensor = load_data("qm7-power-spectrum.npz")
    reduced_tensor = reduce_over_samples(tensor)
    assert isinstance(reduced_tensor, torch.ScriptObject)


def test_operation_as_python():
    check_operation(equistore.torch.sum_over_samples)


def test_operation_as_torch_script():
    scripted = torch.jit.script(equistore.torch.sum_over_samples)
    check_operation(scripted)
