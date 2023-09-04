import torch

import metatensor.torch

from .data import load_data


def check_to_cpu(to):
    tensor = load_data("qm7-power-spectrum.npz")
    print(tensor.block(0).gradients_list())
    assert tensor.block(0).values.device.type == "cpu"
    tensor_to = to(tensor, device="cpu")
    assert tensor_to.block(0).values.device.type == "cpu"


def check_to_dtype(to):
    tensor = load_data("qm7-power-spectrum.npz")
    assert tensor.block(0).values.dtype == torch.float64
    tensor_to = to(tensor, dtype=torch.float32)
    assert tensor_to.block(0).values.dtype == torch.float32
    assert tensor_to.block(0).gradient("positions").values.dtype == torch.float32


def test_operations_as_python():
    check_to_cpu(metatensor.torch.to)
    check_to_dtype(metatensor.torch.to)


def test_operations_as_torch_script():
    check_to_cpu(torch.jit.script(metatensor.torch.to))
    check_to_dtype(torch.jit.script(metatensor.torch.to))
