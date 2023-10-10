import torch

import metatensor.torch

from .data import load_data


def check_operation(reduce_over_samples):
    tensor = load_data("qm7-power-spectrum.npz")

    assert tensor.sample_names == ["structure", "center"]
    reduced_tensor = reduce_over_samples(tensor, "center")

    assert isinstance(reduced_tensor, torch.ScriptObject)
    assert reduced_tensor.sample_names == ["structure"]


def test_operations_as_python():
    check_operation(metatensor.torch.sum_over_samples)
    check_operation(metatensor.torch.mean_over_samples)
    check_operation(metatensor.torch.std_over_samples)
    check_operation(metatensor.torch.var_over_samples)


def test_operations_as_torch_script():
    check_operation(torch.jit.script(metatensor.torch.sum_over_samples))
    check_operation(torch.jit.script(metatensor.torch.mean_over_samples))
    check_operation(torch.jit.script(metatensor.torch.std_over_samples))
    check_operation(torch.jit.script(metatensor.torch.var_over_samples))
