import torch

import equistore.torch

from .data import load_data


def test_operation_as_python():
    tensor = load_data("qm7-power-spectrum.npz")
    random__tensor = equistore.torch.random_uniform_like(tensor)
    assert equistore.torch.equal_metadata(random__tensor, tensor)


def test_operation_as_torch_script():
    tensor = load_data("qm7-power-spectrum.npz")
    random_tensor = torch.jit.script(equistore.torch.random_uniform_like)(tensor)
    assert equistore.torch.equal_metadata(random_tensor, tensor)
