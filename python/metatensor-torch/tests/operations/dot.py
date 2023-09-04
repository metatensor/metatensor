import torch
from packaging import version

import metatensor.torch

from .data import load_data


def check_operation(dot):
    tensor = load_data("qm7-power-spectrum.npz")
    dot_tensor = dot(
        tensor, metatensor.torch.remove_gradients(tensor, ["positions", "cell"])
    )

    # right output type
    assert isinstance(dot_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert dot_tensor._type().name() == "TensorMap"

    # right metadata
    for key in tensor.keys:
        assert dot_tensor.block(key).samples == tensor.block(key).samples
        assert dot_tensor.block(key).properties == tensor.block(key).samples

    # right values
    for key in tensor.keys:
        assert torch.allclose(
            dot_tensor.block(key).values,
            tensor.block(key).values @ tensor.block(key).values.T,
        )


def test_operation_as_python():
    check_operation(metatensor.torch.dot)


def test_operation_as_torch_script():
    check_operation(torch.jit.script(metatensor.torch.dot))
