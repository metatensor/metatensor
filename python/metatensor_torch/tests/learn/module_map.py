import io

import pytest
import torch

from metatensor.torch import Labels, allclose_raise
from metatensor.torch.learn.nn import ModuleMap

from . import _tests_utils


@pytest.fixture
def tensor():
    return _tests_utils.random_single_block_no_components_tensor_map()


try:
    if torch.cuda.is_available():
        HAS_CUDA = True
    else:
        HAS_CUDA = False
except ImportError:
    HAS_CUDA = False


class MockModule(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self._linear = torch.nn.Linear(in_features, out_features)
        self._activation = torch.nn.Sigmoid()
        self._last_layer = torch.nn.Linear(out_features, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._last_layer(self._activation(self._linear(input)))


@pytest.mark.parametrize(
    "out_properties", [None, [Labels(["a", "b"], torch.tensor([[1, 1]]))]]
)
def test_module_map(tensor, out_properties):
    modules = []
    for key in tensor.keys:
        modules.append(
            MockModule(
                in_features=len(tensor.block(key).properties),
                out_features=5,
            )
        )

    tensor_module = ModuleMap(tensor.keys, modules, out_properties=out_properties)
    output = tensor_module(tensor)

    for i, item in enumerate(tensor.items()):
        key, block = item
        module = modules[i]
        assert tensor_module.get_module(key) is module, (
            "modules should be initialized in the same order as keys"
        )

        with torch.no_grad():
            ref_values = module(block.values)
        out_block = output.block(key)
        assert torch.allclose(ref_values, out_block.values)
        if out_properties is None:
            assert out_block.properties == Labels.range("_", len(out_block.properties))
        else:
            assert out_block.properties == out_properties[0]

        for parameter, gradient in block.gradients():
            with torch.no_grad():
                ref_gradient_values = module(gradient.values)
            assert torch.allclose(
                ref_gradient_values, out_block.gradient(parameter).values
            )


@pytest.mark.parametrize("torch_script", [True, False])
def test_to_device(tensor, torch_script):
    devices = ["cpu"]
    if _tests_utils.can_use_mps_backend():
        devices.append("mps")

    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        modules = []
        for key in tensor.keys:
            modules.append(
                MockModule(
                    in_features=len(tensor.block(key).properties),
                    out_features=5,
                )
            )
        module = ModuleMap(
            tensor.keys,
            modules,
            out_properties=[Labels(["a", "b"], torch.tensor([[1, 1]]))],
        )

        if torch_script:
            module = torch.jit.script(module)

        assert module._in_keys.device.type == "cpu"
        for label in module._out_properties:
            assert label.device.type == "cpu"

        module.to(device=device)

        # at this point, the parameters should have been moved,
        # but the input keys and output properties should still be on cpu
        assert len(list(module.parameters())) > 0
        for parameter in module.parameters():
            assert parameter.device.type == device

        assert module._in_keys.device.type == "cpu"
        for label in module._out_properties:
            assert label.device.type == "cpu"

        device_tensor = tensor.to(device=device)
        output = module(device_tensor)
        assert output.device.type == device

        # the input keys and output properties should now be on device
        assert module._in_keys.device.type == device
        for label in module._out_properties:
            assert label.device.type == device


def test_torchscript(tensor):
    modules = []
    for key in tensor.keys:
        modules.append(
            MockModule(
                in_features=len(tensor.block(key).properties),
                out_features=5,
            )
        )
    tensor_module = ModuleMap(tensor.keys, modules)
    ref_tensor = tensor_module(tensor)

    tensor_module_script = torch.jit.script(tensor_module)
    out_tensor = tensor_module_script(tensor)

    allclose_raise(ref_tensor, out_tensor)

    # tests if member functions work that do not appear in forward
    tensor_module_script.get_module(tensor.keys[0])

    # test save load
    scripted = torch.jit.script(tensor_module_script)
    with io.BytesIO() as buffer:
        torch.jit.save(scripted, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
