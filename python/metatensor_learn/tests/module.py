import os
import warnings
from typing import Dict, List

import pytest

from metatensor import Labels, TensorBlock, TensorMap


torch = pytest.importorskip("torch")

from metatensor.learn import nn  # noqa: E402

from . import _tests_utils  # noqa: E402


@pytest.fixture(scope="module")
def devices_to_test():
    """Get a list of non-CPU devices available for testing."""

    devices = []
    if _tests_utils.can_use_mps_backend():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    return devices


def _create_block(sample_name):
    return TensorBlock(
        values=torch.rand(3, 4, dtype=torch.float64),
        samples=Labels([sample_name], torch.arange(3).reshape(-1, 1)),
        components=[],
        properties=Labels(["p"], torch.arange(4).reshape(-1, 1)),
    )


def _create_tensor(key_name):
    return TensorMap(
        keys=Labels([key_name], torch.zeros((1, 1), dtype=torch.int32)),
        blocks=[_create_block("test")],
    )


class LabelsModule(nn.Module):
    nested: Dict[str, Dict[int, List[List[Labels]]]]

    def __init__(self, name):
        super().__init__()
        values = torch.arange(2).reshape(-1, 1)
        labels_value = Labels([name], values)
        container_value = {"labels": Labels([name], values)}
        list_value = [Labels([name], values)]
        tuple_value = tuple([Labels([name], values)])
        nested_value = {
            "dict": {42: [[Labels([name], values)], []], 50: []},
            "empty": {},
        }

        # registered via explicit register_buffer
        self.register_buffer("labels", labels_value)
        self.register_buffer("nested", nested_value)
        # registered via Buffer wrapper
        self.dict = nn.Buffer(container_value)
        self.list = nn.Buffer(list_value)
        self.tuple = nn.Buffer(tuple_value)

        # unregistered — should NOT be tracked
        self.unregistered_labels = Labels([name], values)
        self.unregistered_list = [Labels([name], values)]


class BlockModule(nn.Module):
    nested: Dict[str, Dict[int, List[List[TensorBlock]]]]

    def __init__(self, name):
        super().__init__()
        block_value = _create_block(name)
        container_value = {"block": _create_block(name)}
        list_value = [_create_block(name)]
        tuple_value = tuple([_create_block(name)])
        nested_value = {
            "dict": {42: [[_create_block(name)], []], 50: []},
            "empty": {},
        }

        # registered via explicit register_buffer
        self.register_buffer("block", block_value)
        self.register_buffer("nested", nested_value)
        # registered via Buffer wrapper
        self.dict = nn.Buffer(container_value)
        self.list = nn.Buffer(list_value)
        self.tuple = nn.Buffer(tuple_value)

        # unregistered — should NOT be tracked
        self.unregistered_block = _create_block(name)
        self.unregistered_list = [_create_block(name)]


class TensorModule(nn.Module):
    nested: Dict[str, Dict[int, List[List[TensorMap]]]]

    def __init__(self, name):
        super().__init__()
        tensor_value = _create_tensor(name)
        container_value = {"tensor": _create_tensor(name)}
        list_value = [_create_tensor(name)]
        tuple_value = tuple([_create_tensor(name)])
        nested_value = {
            "dict": {42: [[_create_tensor(name)], []], 50: []},
            "empty": {},
        }

        # registered via explicit register_buffer
        self.register_buffer("tensor", tensor_value)
        self.register_buffer("nested", nested_value)
        # registered via Buffer wrapper
        self.dict = nn.Buffer(container_value)
        self.list = nn.Buffer(list_value)
        self.tuple = nn.Buffer(tuple_value)

        # unregistered — should NOT be tracked
        self.unregistered_tensor = _create_tensor(name)
        self.unregistered_list = [_create_tensor(name)]


class EverythingModule(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.register_buffer("labels", Labels([name], torch.arange(2).reshape(-1, 1)))
        self.register_buffer("block", _create_block(name))
        self.register_buffer("tensor", _create_tensor(name))
        self.register_buffer(
            "tuple",
            (
                Labels([name], torch.arange(2).reshape(-1, 1)),
                _create_block(name),
                _create_tensor(name),
            ),
        )

        # unregistered metatensor data — should NOT be moved by .to()
        self.unregistered_labels = Labels([name], torch.arange(2).reshape(-1, 1))
        self.unregistered_block = _create_block(name)
        self.unregistered_tensor = _create_tensor(name)

        # check nested modules
        self.a = LabelsModule(name)
        self.b = BlockModule(name)
        self.c = TensorModule(name)


def test_to(devices_to_test):
    def check_device_dtype(module, device, dtype):
        # registered: should have moved
        assert module.labels.device.type == device

        assert module.block.device.type == device
        assert module.block.dtype == dtype

        assert module.tensor.device.type == device
        assert module.tensor.dtype == dtype

        assert module.tuple[0].device.type == device
        assert module.tuple[1].device.type == device
        assert module.tuple[1].dtype == dtype
        assert module.tuple[2].device.type == device
        assert module.tuple[2].dtype == dtype

        assert module.a.labels.device.type == device
        assert module.a.dict["labels"].device.type == device
        assert module.a.list[0].device.type == device
        assert module.a.nested["dict"][42][0][0].device.type == device

        assert module.b.block.device.type == device
        assert module.b.block.dtype == dtype
        assert module.b.dict["block"].device.type == device
        assert module.b.dict["block"].dtype == dtype
        assert module.b.list[0].device.type == device
        assert module.b.list[0].dtype == dtype
        assert module.b.nested["dict"][42][0][0].device.type == device
        assert module.b.nested["dict"][42][0][0].dtype == dtype

        assert module.c.tensor.device.type == device
        assert module.c.tensor.dtype == dtype
        assert module.c.dict["tensor"].device.type == device
        assert module.c.dict["tensor"].dtype == dtype
        assert module.c.list[0].device.type == device
        assert module.c.list[0].dtype == dtype
        assert module.c.nested["dict"][42][0][0].device.type == device
        assert module.c.nested["dict"][42][0][0].dtype == dtype

        # unregistered: should NOT have moved (stays on cpu/float64)
        assert module.unregistered_labels.device.type == "cpu"
        assert module.unregistered_block.device.type == "cpu"
        assert module.unregistered_block.dtype == torch.float64
        assert module.unregistered_tensor.device.type == "cpu"
        assert module.unregistered_tensor.dtype == torch.float64

    module = EverythingModule("test")
    check_device_dtype(module, "cpu", torch.float64)

    module = module.to(dtype=torch.float32)
    check_device_dtype(module, "cpu", torch.float32)

    warnings.filterwarnings(
        "ignore",
        message="Blocks values and keys for this TensorMap are on different devices",
    )
    warnings.filterwarnings(
        "ignore",
        message="Values and labels for this block are on different devices",
    )

    for device in devices_to_test:
        module = module.to(device=device)
        check_device_dtype(module, device, torch.float32)

        # in-place modification also works
        module.cpu()
        check_device_dtype(module, "cpu", torch.float32)

        if device == "cuda":
            module.cuda()
            check_device_dtype(module, device, torch.float32)

    # make sure everything works when the outer module is a standard torch.nn.Module
    class Recursive(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.sub_module = EverythingModule(name)

    module = Recursive("test")
    check_device_dtype(module.sub_module, "cpu", torch.float64)

    module = module.to(dtype=torch.float32)
    check_device_dtype(module.sub_module, "cpu", torch.float32)

    for device in devices_to_test:
        module = module.to(device=device)
        check_device_dtype(module.sub_module, device, torch.float32)

        # in-place modification also works
        module.cpu()
        check_device_dtype(module.sub_module, "cpu", torch.float32)

        if device == "cuda":
            module.cuda()
            check_device_dtype(module.sub_module, device, torch.float32)


def test_state_dict_labels(tmpdir):
    def check_state_dict(entry):
        assert isinstance(entry, tuple)
        assert entry[0] == "metatensor.Labels"
        assert isinstance(entry[1], bytes)
        assert isinstance(entry[2], torch.Tensor)
        assert len(entry[2]) == 0

        assert entry[2].dtype == torch.int32
        assert entry[2].device.type == "cpu"

    module = LabelsModule("test")

    state_dict = module.state_dict()
    # registered data appears in _extra_state
    check_state_dict(state_dict["_extra_state"]["labels"])
    check_state_dict(state_dict["_extra_state"]["dict"]["labels"])
    check_state_dict(state_dict["_extra_state"]["list"][0])
    check_state_dict(state_dict["_extra_state"]["tuple"][0])
    check_state_dict(state_dict["_extra_state"]["nested"]["dict"][42][0][0])
    # unregistered data does NOT appear in _extra_state
    assert "unregistered_labels" not in state_dict["_extra_state"]
    assert "unregistered_list" not in state_dict["_extra_state"]

    module = LabelsModule("something")
    assert module.labels.names == ["something"]
    assert module.dict["labels"].names == ["something"]
    assert module.list[0].names == ["something"]
    assert module.tuple[0].names == ["something"]
    assert module.nested["dict"][42][0][0].names == ["something"]
    # unregistered retains its initial value
    assert module.unregistered_labels.names == ["something"]

    module.load_state_dict(state_dict)
    assert module.labels.names == ["test"]
    assert module.dict["labels"].names == ["test"]
    assert module.list[0].names == ["test"]
    assert module.tuple[0].names == ["test"]
    assert module.nested["dict"][42][0][0].names == ["test"]
    # unregistered still has its initial value (not overwritten)
    assert module.unregistered_labels.names == ["something"]

    # make sure everything works when the outer module is a standard torch.nn.Module
    class Recursive(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.sub_module = LabelsModule(name)

    module = Recursive("test")
    state_dict = module.state_dict()
    check_state_dict(state_dict["sub_module._extra_state"]["labels"])
    check_state_dict(state_dict["sub_module._extra_state"]["dict"]["labels"])
    check_state_dict(state_dict["sub_module._extra_state"]["list"][0])
    check_state_dict(state_dict["sub_module._extra_state"]["tuple"][0])
    check_state_dict(state_dict["sub_module._extra_state"]["nested"]["dict"][42][0][0])
    assert "unregistered_labels" not in state_dict["sub_module._extra_state"]
    assert "unregistered_list" not in state_dict["sub_module._extra_state"]

    module = Recursive("something")
    assert module.sub_module.labels.names == ["something"]
    assert module.sub_module.dict["labels"].names == ["something"]
    assert module.sub_module.list[0].names == ["something"]
    assert module.sub_module.tuple[0].names == ["something"]
    assert module.sub_module.nested["dict"][42][0][0].names == ["something"]

    module.load_state_dict(state_dict)
    assert module.sub_module.labels.names == ["test"]
    assert module.sub_module.dict["labels"].names == ["test"]
    assert module.sub_module.list[0].names == ["test"]
    assert module.sub_module.tuple[0].names == ["test"]
    assert module.sub_module.nested["dict"][42][0][0].names == ["test"]


def test_state_dict_block():
    def check_state_dict(entry):
        assert isinstance(entry, tuple)
        assert entry[0] == "metatensor.TensorBlock"
        assert isinstance(entry[1], bytes)
        assert isinstance(entry[2], torch.Tensor)
        assert len(entry[2]) == 0

        assert entry[2].dtype == torch.float64
        assert entry[2].device.type == "cpu"

    module = BlockModule("test")

    state_dict = module.state_dict()
    check_state_dict(state_dict["_extra_state"]["block"])
    check_state_dict(state_dict["_extra_state"]["dict"]["block"])
    check_state_dict(state_dict["_extra_state"]["list"][0])
    check_state_dict(state_dict["_extra_state"]["tuple"][0])
    check_state_dict(state_dict["_extra_state"]["nested"]["dict"][42][0][0])
    assert "unregistered_block" not in state_dict["_extra_state"]
    assert "unregistered_list" not in state_dict["_extra_state"]

    module = BlockModule("something")
    assert module.block.samples.names == ["something"]
    assert module.dict["block"].samples.names == ["something"]
    assert module.list[0].samples.names == ["something"]
    assert module.tuple[0].samples.names == ["something"]
    assert module.nested["dict"][42][0][0].samples.names == ["something"]

    module.load_state_dict(state_dict)
    assert module.block.samples.names == ["test"]
    assert module.dict["block"].samples.names == ["test"]
    assert module.list[0].samples.names == ["test"]
    assert module.tuple[0].samples.names == ["test"]
    assert module.nested["dict"][42][0][0].samples.names == ["test"]

    # make sure everything works when the outer module is a standard torch.nn.Module
    class Recursive(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.sub_module = BlockModule(name)

    module = Recursive("test")
    state_dict = module.state_dict()
    check_state_dict(state_dict["sub_module._extra_state"]["block"])
    check_state_dict(state_dict["sub_module._extra_state"]["dict"]["block"])
    check_state_dict(state_dict["sub_module._extra_state"]["list"][0])
    check_state_dict(state_dict["sub_module._extra_state"]["tuple"][0])
    check_state_dict(state_dict["sub_module._extra_state"]["nested"]["dict"][42][0][0])

    module = Recursive("something")
    assert module.sub_module.block.samples.names == ["something"]
    assert module.sub_module.dict["block"].samples.names == ["something"]
    assert module.sub_module.list[0].samples.names == ["something"]
    assert module.sub_module.tuple[0].samples.names == ["something"]
    assert module.sub_module.nested["dict"][42][0][0].samples.names == ["something"]

    module.load_state_dict(state_dict)
    assert module.sub_module.block.samples.names == ["test"]
    assert module.sub_module.dict["block"].samples.names == ["test"]
    assert module.sub_module.list[0].samples.names == ["test"]
    assert module.sub_module.tuple[0].samples.names == ["test"]
    assert module.sub_module.nested["dict"][42][0][0].samples.names == ["test"]


def _check_serialized_dtype(value, dtype):
    """Check the dtype of serialized metatensor data"""
    assert value[2].dtype == dtype


def _check_serialized_device(value, device):
    """Check the device of serialized metatensor data"""
    assert value[2].device.type == device


def test_state_dict_block_device_dtype(tmpdir, devices_to_test):
    # Test loading from a different dtype
    module = BlockModule("test")
    module = module.to(dtype=torch.float32)

    state_dict = module.state_dict()
    extra_state = state_dict["_extra_state"]
    # check that the state dict records the correct dtype
    _check_serialized_dtype(extra_state["block"], torch.float32)
    _check_serialized_dtype(extra_state["dict"]["block"], torch.float32)
    _check_serialized_dtype(extra_state["list"][0], torch.float32)
    _check_serialized_dtype(extra_state["tuple"][0], torch.float32)
    _check_serialized_dtype(extra_state["nested"]["dict"][42][0][0], torch.float32)
    # unregistered still not in extra_state
    assert "unregistered_block" not in extra_state

    # check that loading the state dict creates data on the correct dtype
    module = BlockModule("another")
    assert module.block.dtype == torch.float64
    assert module.dict["block"].dtype == torch.float64
    assert module.list[0].dtype == torch.float64
    assert module.tuple[0].dtype == torch.float64
    assert module.nested["dict"][42][0][0].dtype == torch.float64

    module.load_state_dict(state_dict)
    assert module.block.dtype == torch.float32
    assert module.dict["block"].dtype == torch.float32
    assert module.list[0].dtype == torch.float32
    assert module.tuple[0].dtype == torch.float32
    assert module.nested["dict"][42][0][0].dtype == torch.float32

    warnings.filterwarnings(
        "ignore", message="Values and labels for this block are on different devices"
    )

    # Test loading from a different device
    for device in devices_to_test:
        module = BlockModule("test")
        module = module.to(dtype=torch.float32)
        module = module.to(device=device)

        state_dict = module.state_dict()
        extra_state = state_dict["_extra_state"]
        # check that the state dict records the correct device
        _check_serialized_device(extra_state["block"], device)
        _check_serialized_device(extra_state["dict"]["block"], device)
        _check_serialized_device(extra_state["list"][0], device)
        _check_serialized_device(extra_state["tuple"][0], device)
        _check_serialized_device(extra_state["nested"]["dict"][42][0][0], device)

        # check that loading the state dict creates data on the correct device
        module = BlockModule("another")
        assert module.block.device.type == "cpu"
        assert module.dict["block"].device.type == "cpu"
        assert module.list[0].device.type == "cpu"
        assert module.tuple[0].device.type == "cpu"
        assert module.nested["dict"][42][0][0].device.type == "cpu"

        module.load_state_dict(state_dict)
        assert module.block.device.type == device
        assert module.dict["block"].device.type == device
        assert module.list[0].device.type == device
        assert module.tuple[0].device.type == device
        assert module.nested["dict"][42][0][0].device.type == device

        # check that loading the state dict from file also works
        path = os.path.join(tmpdir, "data.pt")
        torch.save(state_dict, path)

        state_dict = torch.load(path, map_location=device)

        module = BlockModule("another")
        assert module.block.device.type == "cpu"

        module.load_state_dict(state_dict)
        assert module.block.device.type == device
        assert module.dict["block"].device.type == device
        assert module.list[0].device.type == device
        assert module.tuple[0].device.type == device
        assert module.nested["dict"][42][0][0].device.type == device


def test_state_dict_tensor(tmpdir):
    def check_state_dict(entry):
        assert isinstance(entry, tuple)
        assert entry[0] == "metatensor.TensorMap"
        assert isinstance(entry[1], bytes)
        assert isinstance(entry[2], torch.Tensor)
        assert len(entry[2]) == 0

        assert entry[2].dtype == torch.float64
        assert entry[2].device.type == "cpu"

    module = TensorModule("test")

    state_dict = module.state_dict()
    extra_state = state_dict["_extra_state"]
    check_state_dict(extra_state["tensor"])
    check_state_dict(extra_state["dict"]["tensor"])
    check_state_dict(extra_state["list"][0])
    check_state_dict(extra_state["tuple"][0])
    check_state_dict(extra_state["nested"]["dict"][42][0][0])
    assert "unregistered_tensor" not in extra_state
    assert "unregistered_list" not in extra_state

    module = TensorModule("something")
    assert module.tensor.keys.names == ["something"]
    assert module.dict["tensor"].keys.names == ["something"]
    assert module.list[0].keys.names == ["something"]
    assert module.tensor.keys.names == ["something"]
    assert module.tensor.keys.names == ["something"]

    module.load_state_dict(state_dict)
    assert module.tensor.keys.names == ["test"]
    assert module.dict["tensor"].keys.names == ["test"]
    assert module.list[0].keys.names == ["test"]
    assert module.tensor.keys.names == ["test"]
    assert module.tensor.keys.names == ["test"]

    # make sure everything works when the outer module is a standard torch.nn.Module
    class Recursive(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.sub_module = TensorModule(name)

    module = Recursive("test")
    state_dict = module.state_dict()
    extra_state = state_dict["sub_module._extra_state"]
    check_state_dict(extra_state["tensor"])
    check_state_dict(extra_state["dict"]["tensor"])
    check_state_dict(extra_state["list"][0])
    check_state_dict(extra_state["tuple"][0])
    check_state_dict(extra_state["nested"]["dict"][42][0][0])

    module = Recursive("something")
    assert module.sub_module.tensor.keys.names == ["something"]
    assert module.sub_module.dict["tensor"].keys.names == ["something"]
    assert module.sub_module.list[0].keys.names == ["something"]
    assert module.sub_module.tensor.keys.names == ["something"]
    assert module.sub_module.tensor.keys.names == ["something"]

    module.load_state_dict(state_dict)
    assert module.sub_module.tensor.keys.names == ["test"]
    assert module.sub_module.dict["tensor"].keys.names == ["test"]
    assert module.sub_module.list[0].keys.names == ["test"]
    assert module.sub_module.tensor.keys.names == ["test"]
    assert module.sub_module.tensor.keys.names == ["test"]


def test_state_dict_tensor_device_dtype(tmpdir, devices_to_test):
    # Test loading from a different dtype
    module = TensorModule("test")
    module = module.to(dtype=torch.float32)

    state_dict = module.state_dict()
    extra_state = state_dict["_extra_state"]
    # check that the state dict records the correct dtype
    _check_serialized_dtype(extra_state["tensor"], torch.float32)
    _check_serialized_dtype(extra_state["dict"]["tensor"], torch.float32)
    _check_serialized_dtype(extra_state["list"][0], torch.float32)
    _check_serialized_dtype(extra_state["tuple"][0], torch.float32)
    _check_serialized_dtype(extra_state["nested"]["dict"][42][0][0], torch.float32)

    # check that loading the state dict creates data on the correct dtype
    module = TensorModule("another")
    assert module.tensor.dtype == torch.float64
    assert module.dict["tensor"].dtype == torch.float64
    assert module.list[0].dtype == torch.float64
    assert module.tuple[0].dtype == torch.float64
    assert module.nested["dict"][42][0][0].dtype == torch.float64

    module.load_state_dict(state_dict)
    assert module.tensor.dtype == torch.float32
    assert module.dict["tensor"].dtype == torch.float32
    assert module.list[0].dtype == torch.float32
    assert module.tuple[0].dtype == torch.float32
    assert module.nested["dict"][42][0][0].dtype == torch.float32

    warnings.filterwarnings(
        "ignore",
        message="Blocks values and keys for this TensorMap are on different devices",
    )

    # Test loading from a different device
    for device in devices_to_test:
        module = TensorModule("test")
        module = module.to(dtype=torch.float32)
        module = module.to(device=device)

        state_dict = module.state_dict()
        extra_state = state_dict["_extra_state"]
        # check that the state dict records the correct device
        _check_serialized_device(extra_state["tensor"], device)
        _check_serialized_device(extra_state["dict"]["tensor"], device)
        _check_serialized_device(extra_state["list"][0], device)
        _check_serialized_device(extra_state["tuple"][0], device)
        _check_serialized_device(extra_state["nested"]["dict"][42][0][0], device)

        # check that loading the state dict creates data on the correct device
        module = TensorModule("another")
        assert module.tensor.device.type == "cpu"
        assert module.dict["tensor"].device.type == "cpu"
        assert module.list[0].device.type == "cpu"
        assert module.tuple[0].device.type == "cpu"
        assert module.nested["dict"][42][0][0].device.type == "cpu"

        module.load_state_dict(state_dict)
        assert module.tensor.device.type == device
        assert module.dict["tensor"].device.type == device
        assert module.list[0].device.type == device
        assert module.tuple[0].device.type == device
        assert module.nested["dict"][42][0][0].device.type == device

        # check that loading the state dict from file also works
        path = os.path.join(tmpdir, "data.pt")
        torch.save(state_dict, path)

        state_dict = torch.load(path, map_location=device)

        module = TensorModule("another")
        assert module.tensor.device.type == "cpu"

        module.load_state_dict(state_dict)
        assert module.tensor.device.type == device
        assert module.dict["tensor"].device.type == device
        assert module.list[0].device.type == device
        assert module.tuple[0].device.type == device
        assert module.nested["dict"][42][0][0].device.type == device


def test_non_persistent_buffer():
    # Check that metatensor buffers registered with persistent=False are excluded
    # from state_dict, but still moved by .to()
    class NonPersistentModule(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.register_buffer(
                "persistent_labels",
                Labels([name], torch.arange(2).reshape(-1, 1)),
            )
            self.register_buffer(
                "non_persistent_labels",
                Labels([name], torch.arange(2).reshape(-1, 1)),
                persistent=False,
            )

    module = NonPersistentModule("test")
    state_dict = module.state_dict()

    # persistent buffer is in state_dict
    assert "persistent_labels" in state_dict["_extra_state"]
    # non-persistent buffer is NOT in state_dict
    assert "non_persistent_labels" not in state_dict["_extra_state"]

    # both buffers should still be moved by .to()
    module = module.to(dtype=torch.float32)
    # Labels don't have dtype, but device should be accessible
    assert module.persistent_labels.device.type == "cpu"
    assert module.non_persistent_labels.device.type == "cpu"

    # loading a state dict should only restore the persistent buffer
    module = NonPersistentModule("something")
    assert module.persistent_labels.names == ["something"]
    assert module.non_persistent_labels.names == ["something"]

    module.load_state_dict(state_dict)
    assert module.persistent_labels.names == ["test"]
    # non-persistent buffer is not restored
    assert module.non_persistent_labels.names == ["something"]
