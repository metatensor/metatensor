import numpy as np
import pytest

from metatensor import Labels


torch = pytest.importorskip("torch")

from metatensor.learn.nn import ModuleMap  # noqa: E402

from . import _tests_utils  # noqa: E402


@pytest.fixture
def tensor():
    """
    random tensor map with no components using torch as array backend
    """
    return _tests_utils.random_single_block_no_components_tensor_map(use_torch=True)


class MockModule(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self._linear = torch.nn.Linear(in_features, out_features)
        self._activation = torch.nn.Sigmoid()
        self._last_layer = torch.nn.Linear(out_features, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._last_layer(self._activation(self._linear(input)))


@pytest.mark.parametrize(
    "out_properties", [None, [Labels(["a", "b"], np.array([[1, 1]]))]]
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
    out_tensor = tensor_module(tensor)

    for i, item in enumerate(tensor.items()):
        key, block = item
        module = modules[i]
        assert tensor_module.get_module(key) is module, (
            "modules should be initialized in the same order as keys"
        )

        ref_values = module(block.values)
        out_block = out_tensor.block(key)
        assert torch.allclose(ref_values, out_block.values)
        if out_properties is None:
            assert out_block.properties == Labels.range("_", len(out_block.properties))
        else:
            assert out_block.properties == out_properties[0]

        for parameter, gradient in block.gradients():
            ref_gradient_values = module(gradient.values)
            assert torch.allclose(
                ref_gradient_values, out_block.gradient(parameter).values
            )


@pytest.mark.filterwarnings("ignore:.*If you are using PyTorch.*")
def test_to_device(tensor):
    """
    Checks the `to` function of module map by moving the module to another device and
    checking that the output tensor is on this device.
    """
    devices = ["meta"]
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
            out_properties=[Labels(["a", "b"], np.array([[1, 1]]))],
        )

        assert module._in_keys.device == "cpu"
        for label in module._out_properties:
            assert label.device == "cpu"

        module.to(device=device)

        # at this point, the parameters should have been moved,
        # but the input keys and output properties should still be on cpu
        assert len(list(module.parameters())) > 0
        for parameter in module.parameters():
            assert parameter.device.type == device

        assert module._in_keys.device == "cpu"
        for label in module._out_properties:
            assert label.device == "cpu"

        device_tensor = tensor.to(device=device)
        output = module(device_tensor)

        assert output.device.type == device

        # the input keys and output properties are still on cpu, because
        # metatensor.Labels are never moved to a different device (by opposition to
        # `metatensor.torch.Labels` which can be moved to another device)
        assert module._in_keys.device == "cpu"
        for label in module._out_properties:
            assert label.device == "cpu"


def test_to_dtype(tensor):
    modules = []
    for key in tensor.keys:
        modules.append(
            MockModule(
                in_features=len(tensor.block(key).properties),
                out_features=5,
            )
        )

    tensor_module = ModuleMap(tensor.keys, modules)
    output = tensor_module(tensor)
    assert output.dtype == tensor.dtype

    tensor_module = tensor_module.to(torch.float64)
    tensor = tensor.to(dtype=torch.float64)
    output = tensor_module(tensor)
    assert output.dtype == tensor.dtype
