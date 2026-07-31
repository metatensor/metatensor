import pytest

from metatensor import Labels


torch = pytest.importorskip("torch")

from metatensor.learn.nn import ModuleMap  # noqa: E402

from . import _tests_utils  # noqa: E402
from ._tests_utils import check_labels_state_dict  # noqa: E402


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
    "out_properties", [None, [Labels(["a", "b"], torch.tensor([[1, 1]]))]]
)
def test_module_map(tensor, out_properties):
    assert isinstance(tensor.keys.values, torch.Tensor)

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
    assert isinstance(tensor.keys.values, torch.Tensor)

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
            out_properties=[Labels(["a", "b"], torch.tensor([[1, 1]]))],
        )

        assert module.in_keys.device.type == "cpu"
        for label in module.out_properties:
            assert label.device.type == "cpu"

        module.to(device=device)

        # at this point, the parameters should have been moved, but the input keys and
        # output properties will still be on cpu, since we are using the metatensor
        # backend, not metatensor.torch.
        assert len(list(module.parameters())) > 0
        for parameter in module.parameters():
            assert parameter.device.type == device

        assert module.in_keys.device.type == device
        for label in module.out_properties:
            assert label.device.type == device


def test_state_dict_structure():
    """
    Checks the exact structure (keys + metatensor tags) of ``ModuleMap.state_dict()``
    to guard against regressions in registered parameters, buffers, and metatensor
    extra state.
    """
    keys = Labels(
        names=["o3_lambda", "o3_sigma"], values=torch.tensor([[0, 1], [1, 1]])
    )
    out_props = [
        Labels(["p"], torch.tensor([[0], [1]])),
        Labels(["p"], torch.tensor([[0], [1]])),
    ]

    # --- with out_properties, mixed bias ---
    module = ModuleMap(
        keys,
        [torch.nn.Linear(3, 2, bias=True), torch.nn.Linear(3, 2, bias=False)],
        out_properties=out_props,
    )
    state_dict = module.state_dict()
    assert set(state_dict.keys()) == {
        "_mts_helper",
        "module_list.0.bias",
        "module_list.0.weight",
        "module_list.1.weight",
        "_extra_state",
    }
    assert set(state_dict["_extra_state"].keys()) == {"_in_keys", "_out_properties"}
    check_labels_state_dict(state_dict["_extra_state"]["_in_keys"])
    out_properties = state_dict["_extra_state"]["_out_properties"]
    assert isinstance(out_properties, list)
    assert len(out_properties) == 2
    for entry in out_properties:
        check_labels_state_dict(entry)

    # --- without out_properties ---
    module = ModuleMap(
        keys,
        [torch.nn.Linear(3, 2, bias=True), torch.nn.Linear(3, 2, bias=False)],
    )
    state_dict = module.state_dict()
    assert set(state_dict.keys()) == {
        "_mts_helper",
        "module_list.0.bias",
        "module_list.0.weight",
        "module_list.1.weight",
        "_extra_state",
    }
    assert set(state_dict["_extra_state"].keys()) == {"_in_keys"}
    check_labels_state_dict(state_dict["_extra_state"]["_in_keys"])


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
