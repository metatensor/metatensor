import torch
from packaging import version

import metatensor.torch

from .data import load_data


def check_to_cpu(to):
    tensor = load_data("qm7-power-spectrum.npz")
    assert tensor.block(0).values.device.type == "cpu"
    tensor_to = to(tensor, device="cpu")

    assert isinstance(tensor_to, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor_to._type().name() == "TensorMap"

    assert tensor_to.block(0).values.device.type == "cpu"


def check_to_cuda(to):
    tensor = load_data("qm7-power-spectrum.npz")
    assert tensor.block(0).values.device.type == "cpu"
    tensor_to = to(tensor, device="cuda")

    assert isinstance(tensor_to, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor_to._type().name() == "TensorMap"

    assert tensor_to.keys.values.device.type == "cuda"
    for block in tensor_to.blocks():
        assert block.samples.values.device.type == "cuda"
        for component in block.components:
            assert component.values.device.type == "cuda"
        assert block.properties.values.device.type == "cuda"
        assert block.values.device.type == "cuda"
        for _, gradient in block.gradients():
            assert gradient.samples.values.device.type == "cuda"
            for component in gradient.components:
                assert component.values.device.type == "cuda"
            assert gradient.properties.values.device.type == "cuda"
            assert gradient.values.device.type == "cuda"


def check_to_dtype(to):
    tensor = load_data("qm7-power-spectrum.npz")
    assert tensor.block(0).values.dtype == torch.float64
    tensor_to = to(tensor, dtype=torch.float32)

    assert isinstance(tensor_to, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor_to._type().name() == "TensorMap"

    assert tensor_to.block(0).values.dtype == torch.float32
    assert tensor_to.block(0).gradient("positions").values.dtype == torch.float32


def test_operations_as_python():
    check_to_cpu(metatensor.torch.to)
    check_to_dtype(metatensor.torch.to)
    if torch.cuda.is_available():
        check_to_cuda(metatensor.torch.to)


def test_operations_as_torch_script():
    check_to_cpu(torch.jit.script(metatensor.torch.to))
    check_to_dtype(torch.jit.script(metatensor.torch.to))
    if torch.cuda.is_available():
        check_to_cuda(torch.jit.script(metatensor.torch.to))
