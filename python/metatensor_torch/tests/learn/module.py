import os
from typing import Dict, List

import pytest
import torch

from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.learn import nn

from . import _tests_utils


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

    def forward(self, x: int) -> Labels:
        if x == 0:
            return self.labels
        elif x == 1:
            return self.dict["labels"]
        elif x == 2:
            return self.list[0]
        elif x == 3:
            return self.tuple[0]
        elif x == 4:
            return self.nested["dict"][42][0][0]
        elif x == 5:
            return self.unregistered_labels
        else:
            return self.labels


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

    def forward(self, x: int) -> TensorBlock:
        if x == 0:
            return self.block
        elif x == 1:
            return self.dict["block"]
        elif x == 2:
            return self.list[0]
        elif x == 3:
            return self.tuple[0]
        elif x == 4:
            return self.nested["dict"][42][0][0]
        elif x == 5:
            return self.unregistered_block
        else:
            return self.block


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

    def forward(self, x: int) -> TensorMap:
        if x == 0:
            return self.tensor
        elif x == 1:
            return self.dict["tensor"]
        elif x == 2:
            return self.list[0]
        elif x == 3:
            return self.tuple[0]
        elif x == 4:
            return self.nested["dict"][42][0][0]
        elif x == 5:
            return self.unregistered_tensor
        else:
            return self.tensor


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


@pytest.mark.parametrize("scripted", [True, False])
def test_to(scripted, devices_to_test):
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
        assert module.a.tuple[0].device.type == device
        assert module.a.nested["dict"][42][0][0].device.type == device

        assert module.b.block.device.type == device
        assert module.b.block.dtype == dtype
        assert module.b.dict["block"].device.type == device
        assert module.b.dict["block"].dtype == dtype
        assert module.b.list[0].device.type == device
        assert module.b.list[0].dtype == dtype
        assert module.b.tuple[0].device.type == device
        assert module.b.tuple[0].dtype == dtype
        assert module.b.nested["dict"][42][0][0].device.type == device
        assert module.b.nested["dict"][42][0][0].dtype == dtype

        assert module.c.tensor.device.type == device
        assert module.c.tensor.dtype == dtype
        assert module.c.dict["tensor"].device.type == device
        assert module.c.dict["tensor"].dtype == dtype
        assert module.c.list[0].device.type == device
        assert module.c.list[0].dtype == dtype
        assert module.c.tuple[0].device.type == device
        assert module.c.tuple[0].dtype == dtype
        assert module.c.nested["dict"][42][0][0].device.type == device
        assert module.c.nested["dict"][42][0][0].dtype == dtype

        # unregistered: should NOT have moved (stays on cpu/float64)
        assert module.unregistered_labels.device.type == "cpu"
        assert module.unregistered_block.device.type == "cpu"
        assert module.unregistered_block.dtype == dtype_to_int(torch.float64)
        assert module.unregistered_tensor.device.type == "cpu"
        assert module.unregistered_tensor.dtype == dtype_to_int(torch.float64)

    module = EverythingModule("test")
    if scripted:
        module = torch.jit.script(module)

    check_device_dtype(module, "cpu", dtype_to_int(torch.float64))

    module = module.to(dtype=torch.float32)
    check_device_dtype(module, "cpu", dtype_to_int(torch.float32))

    for device in devices_to_test:
        module = module.to(device=device)
        check_device_dtype(module, device, dtype_to_int(torch.float32))

        # in-place modification also works
        module.cpu()
        check_device_dtype(module, "cpu", dtype_to_int(torch.float32))

        if device == "cuda":
            module.cuda()
            check_device_dtype(module, device, dtype_to_int(torch.float32))

    # make sure everything works when the outer module is a standard torch.nn.Module
    class Recursive(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.sub_module = EverythingModule(name)

    module = Recursive("test")
    if scripted:
        module = torch.jit.script(module)

    check_device_dtype(module.sub_module, "cpu", dtype_to_int(torch.float64))

    module = module.to(dtype=torch.float32)
    check_device_dtype(module.sub_module, "cpu", dtype_to_int(torch.float32))

    for device in devices_to_test:
        module = module.to(device=device)
        check_device_dtype(module.sub_module, device, dtype_to_int(torch.float32))

        # in-place modification also works
        module.cpu()
        check_device_dtype(module.sub_module, "cpu", dtype_to_int(torch.float32))

        if device == "cuda":
            module.cuda()
            check_device_dtype(module.sub_module, device, dtype_to_int(torch.float32))


def test_torchscript():
    module = LabelsModule("test")
    scripted = torch.jit.script(module)
    assert scripted(0)._type().name() == "Labels"

    module = BlockModule("test")
    scripted = torch.jit.script(module)
    assert scripted(0)._type().name() == "TensorBlock"

    module = TensorModule("test")
    scripted = torch.jit.script(module)
    assert scripted(0)._type().name() == "TensorMap"


@torch.jit.script
def dtype_to_int(dtype: torch.dtype):
    return dtype


def test_state_dict_labels():
    def check_state_dict(entry):
        assert isinstance(entry, tuple)
        assert entry[0] == "metatensor.Labels"
        assert isinstance(entry[1], torch.Tensor)
        assert entry[1].dtype == torch.uint8
        assert isinstance(entry[2], torch.Tensor)
        assert len(entry[2]) == 0

        assert entry[2].dtype == torch.int32
        assert entry[2].device.type == "cpu"

    module = LabelsModule("test")

    state_dict = module.state_dict()
    # data is saved as a buffer (tensor of uint8)
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

    module.load_state_dict(state_dict)
    assert module.labels.names == ["test"]
    assert module.dict["labels"].names == ["test"]
    assert module.list[0].names == ["test"]
    assert module.tuple[0].names == ["test"]
    assert module.nested["dict"][42][0][0].names == ["test"]

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
        assert isinstance(entry[1], torch.Tensor)
        assert entry[1].dtype == torch.uint8
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


def test_state_dict_block_device_dtype(tmpdir, devices_to_test):
    # Test loading from a different dtype
    module = BlockModule("test")
    module = module.to(dtype=torch.float32)

    state_dict = module.state_dict()
    # check that the state dict records the correct dtype
    assert state_dict["_extra_state"]["block"][2].dtype == torch.float32
    assert state_dict["_extra_state"]["dict"]["block"][2].dtype == torch.float32
    assert state_dict["_extra_state"]["list"][0][2].dtype == torch.float32
    assert state_dict["_extra_state"]["tuple"][0][2].dtype == torch.float32
    assert (
        state_dict["_extra_state"]["nested"]["dict"][42][0][0][2].dtype == torch.float32
    )
    # unregistered still not in extra_state
    assert "unregistered_block" not in state_dict["_extra_state"]

    # check that loading the state dict creates data on the correct dtype
    module = BlockModule("another")
    assert module.block.dtype == dtype_to_int(torch.float64)
    assert module.dict["block"].dtype == dtype_to_int(torch.float64)
    assert module.list[0].dtype == dtype_to_int(torch.float64)
    assert module.tuple[0].dtype == dtype_to_int(torch.float64)
    assert module.nested["dict"][42][0][0].dtype == dtype_to_int(torch.float64)

    module.load_state_dict(state_dict)
    assert module.block.dtype == dtype_to_int(torch.float32)
    assert module.dict["block"].dtype == dtype_to_int(torch.float32)
    assert module.list[0].dtype == dtype_to_int(torch.float32)
    assert module.tuple[0].dtype == dtype_to_int(torch.float32)
    assert module.nested["dict"][42][0][0].dtype == dtype_to_int(torch.float32)

    # Test loading from a different device
    for device in devices_to_test:
        module = BlockModule("test")
        module = module.to(dtype=torch.float32)
        module = module.to(device=device)

        state_dict = module.state_dict()
        # check that the state dict records the correct device
        assert state_dict["_extra_state"]["block"][2].device.type == device
        assert state_dict["_extra_state"]["dict"]["block"][2].device.type == device
        assert state_dict["_extra_state"]["list"][0][2].device.type == device
        assert state_dict["_extra_state"]["tuple"][0][2].device.type == device
        assert (
            state_dict["_extra_state"]["nested"]["dict"][42][0][0][2].device.type
            == device
        )

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


def test_state_dict_tensor():
    def check_state_dict(entry):
        assert isinstance(entry, tuple)
        assert entry[0] == "metatensor.TensorMap"
        assert isinstance(entry[1], torch.Tensor)
        assert entry[1].dtype == torch.uint8
        assert isinstance(entry[2], torch.Tensor)
        assert len(entry[2]) == 0

        assert entry[2].dtype == torch.float64
        assert entry[2].device.type == "cpu"

    module = TensorModule("test")

    state_dict = module.state_dict()
    check_state_dict(state_dict["_extra_state"]["tensor"])
    check_state_dict(state_dict["_extra_state"]["dict"]["tensor"])
    check_state_dict(state_dict["_extra_state"]["list"][0])
    check_state_dict(state_dict["_extra_state"]["tuple"][0])
    check_state_dict(state_dict["_extra_state"]["nested"]["dict"][42][0][0])
    assert "unregistered_tensor" not in state_dict["_extra_state"]
    assert "unregistered_list" not in state_dict["_extra_state"]

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
    check_state_dict(state_dict["sub_module._extra_state"]["tensor"])
    check_state_dict(state_dict["sub_module._extra_state"]["dict"]["tensor"])
    check_state_dict(state_dict["sub_module._extra_state"]["list"][0])
    check_state_dict(state_dict["sub_module._extra_state"]["tuple"][0])
    check_state_dict(state_dict["sub_module._extra_state"]["nested"]["dict"][42][0][0])

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


def test_state_dict_tensorr_device_dtype(tmpdir, devices_to_test):
    # Test loading from a different dtype
    module = TensorModule("test")
    module = module.to(dtype=torch.float32)

    state_dict = module.state_dict()
    # check that the state dict records the correct dtype
    assert state_dict["_extra_state"]["tensor"][2].dtype == torch.float32
    assert state_dict["_extra_state"]["dict"]["tensor"][2].dtype == torch.float32
    assert state_dict["_extra_state"]["list"][0][2].dtype == torch.float32
    assert state_dict["_extra_state"]["tuple"][0][2].dtype == torch.float32
    assert (
        state_dict["_extra_state"]["nested"]["dict"][42][0][0][2].dtype == torch.float32
    )

    # check that loading the state dict creates data on the correct dtype
    module = TensorModule("another")
    assert module.tensor.dtype == dtype_to_int(torch.float64)
    assert module.dict["tensor"].dtype == dtype_to_int(torch.float64)
    assert module.list[0].dtype == dtype_to_int(torch.float64)
    assert module.tuple[0].dtype == dtype_to_int(torch.float64)
    assert module.nested["dict"][42][0][0].dtype == dtype_to_int(torch.float64)

    module.load_state_dict(state_dict)
    assert module.tensor.dtype == dtype_to_int(torch.float32)
    assert module.dict["tensor"].dtype == dtype_to_int(torch.float32)
    assert module.list[0].dtype == dtype_to_int(torch.float32)
    assert module.tuple[0].dtype == dtype_to_int(torch.float32)
    assert module.nested["dict"][42][0][0].dtype == dtype_to_int(torch.float32)

    # Test loading from a different device
    for device in devices_to_test:
        module = TensorModule("test")
        module = module.to(dtype=torch.float32)
        module = module.to(device=device)

        state_dict = module.state_dict()
        # check that the state dict records the correct device
        assert state_dict["_extra_state"]["tensor"][2].device.type == device
        assert state_dict["_extra_state"]["dict"]["tensor"][2].device.type == device
        assert state_dict["_extra_state"]["list"][0][2].device.type == device
        assert state_dict["_extra_state"]["tuple"][0][2].device.type == device
        assert (
            state_dict["_extra_state"]["nested"]["dict"][42][0][0][2].device.type
            == device
        )

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


@pytest.mark.parametrize("scripted", [True, False])
def test_non_persistent_buffer(scripted):
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
    if scripted:
        module = torch.jit.script(module)

    # both buffers should be moved by .to()
    module = module.to(dtype=torch.float32)
    assert module.persistent_labels.device.type == "cpu"
    assert module.non_persistent_labels.device.type == "cpu"

    # check state_dict on the non-scripted module (scripted modules store
    # _extra_state differently)
    module = NonPersistentModule("test")
    state_dict = module.state_dict()

    # persistent buffer is in state_dict
    assert "persistent_labels" in state_dict["_extra_state"]
    # non-persistent buffer is NOT in state_dict
    assert "non_persistent_labels" not in state_dict["_extra_state"]

    # loading a state dict should only restore the persistent buffer
    module = NonPersistentModule("something")
    assert module.persistent_labels.names == ["something"]
    assert module.non_persistent_labels.names == ["something"]

    module.load_state_dict(state_dict)
    assert module.persistent_labels.names == ["test"]
    # non-persistent buffer is not restored
    assert module.non_persistent_labels.names == ["something"]


def test_exisiting_module(devices_to_test):
    def test_device_dtype(module, device, dtype):
        assert module.labels.device.type == device
        assert module.labels.values.dtype == torch.int32
        assert module.block.device.type == device
        assert module.block.dtype == dtype_to_int(dtype)
        assert module.tensor.device.type == device
        assert module.tensor.dtype == dtype_to_int(dtype)

        assert module.tuple[0].device.type == device
        assert module.tuple[0].values.dtype == torch.int32
        assert module.tuple[1].device.type == device
        assert module.tuple[1].dtype == dtype_to_int(dtype)
        assert module.tuple[2].device.type == device
        assert module.tuple[2].dtype == dtype_to_int(dtype)

        assert module.a.labels.device.type == device
        assert module.a.labels.values.dtype == torch.int32
        assert module.a.dict["labels"].device.type == device
        assert module.a.dict["labels"].values.dtype == torch.int32
        assert module.a.list[0].device.type == device
        assert module.a.list[0].values.dtype == torch.int32
        assert module.a.tuple[0].device.type == device
        assert module.a.tuple[0].values.dtype == torch.int32
        assert module.a.nested["dict"][42][0][0].device.type == device
        assert module.a.nested["dict"][42][0][0].values.dtype == torch.int32

        assert module.b.block.device.type == device
        assert module.b.block.dtype == dtype_to_int(dtype)
        assert module.b.dict["block"].device.type == device
        assert module.b.dict["block"].dtype == dtype_to_int(dtype)
        assert module.b.list[0].device.type == device
        assert module.b.list[0].dtype == dtype_to_int(dtype)
        assert module.b.tuple[0].device.type == device
        assert module.b.tuple[0].dtype == dtype_to_int(dtype)
        assert module.b.nested["dict"][42][0][0].device.type == device
        assert module.b.nested["dict"][42][0][0].dtype == dtype_to_int(dtype)

        assert module.c.tensor.device.type == device
        assert module.c.tensor.dtype == dtype_to_int(dtype)
        assert module.c.dict["tensor"].device.type == device
        assert module.c.dict["tensor"].dtype == dtype_to_int(dtype)
        assert module.c.list[0].device.type == device
        assert module.c.list[0].dtype == dtype_to_int(dtype)
        assert module.c.tuple[0].device.type == device
        assert module.c.tuple[0].dtype == dtype_to_int(dtype)
        assert module.c.nested["dict"][42][0][0].device.type == device
        assert module.c.nested["dict"][42][0][0].dtype == dtype_to_int(dtype)

        assert module.d.test.device.type == device
        assert module.d.test.dtype == dtype

        assert module.e.test.device.type == device
        assert module.e.test.dtype == dtype

    # check that already scripted module still work
    ROOT = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )

    module = torch.jit.load(
        os.path.join(ROOT, "metatensor-torch", "tests", "test-module.pt")
    )
    test_device_dtype(module, "cpu", torch.float64)

    module = module.to(dtype=torch.float32)
    test_device_dtype(module, "cpu", torch.float32)

    for device in devices_to_test:
        module = module.to(device=device)
        test_device_dtype(module, device, torch.float32)

        # in-place modification also works
        module.cpu()
        test_device_dtype(module, "cpu", torch.float32)

        if device == "cuda":
            module.cuda()
            test_device_dtype(module, device, torch.float32)
