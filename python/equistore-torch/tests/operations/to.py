import torch

import equistore.torch

from .data import load_data


def check_to_gpu(to):
    tensor = load_data("qm7-power-spectrum.npz")
    assert tensor.block(0).values.device.type == "cpu"


def check_to_dtype(to):
    tensor = load_data("qm7-power-spectrum.npz")
    assert tensor.block(0).values.dtype == torch.float64


def test_operations_as_python():
    check_to_gpu(equistore.torch.to)
    check_to_dtype(equistore.torch.to)


def test_operations_as_torch_script():
    check_to_gpu(torch.jit.script(equistore.torch.to))
    check_to_dtype(torch.jit.script(equistore.torch.to))
