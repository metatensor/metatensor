import torch

import equistore.torch

from .data import load_data


def check_operation(reduce_over_samples):
    tensor = load_data("qm7-power-spectrum.npz")

    assert tensor.sample_names == ["structure", "center"]
    reduced_tensor = reduce_over_samples(tensor, "center")

    assert isinstance(reduced_tensor, torch.ScriptObject)
    assert reduced_tensor.sample_names == ["structure"]


def test_operations_as_python():
    check_operation(equistore.torch.sum_over_samples)
    check_operation(equistore.torch.mean_over_samples)
    check_operation(equistore.torch.std_over_samples)
    check_operation(equistore.torch.var_over_samples)


def test_operations_as_torch_script():
    check_operation(torch.jit.script(equistore.torch.sum_over_samples))
    check_operation(torch.jit.script(equistore.torch.mean_over_samples))
    check_operation(torch.jit.script(equistore.torch.std_over_samples))
    check_operation(torch.jit.script(equistore.torch.var_over_samples))
