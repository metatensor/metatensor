import os
import warnings
from typing import Dict, List

import numpy as np
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
        samples=Labels([sample_name], np.arange(3).reshape(-1, 1)),
        components=[],
        properties=Labels(["p"], np.arange(4).reshape(-1, 1)),
    )


def _create_tensor(key_name):
    return TensorMap(
        keys=Labels([key_name], np.zeros((1, 1), dtype=np.int32)),
        blocks=[_create_block("test")],
    )


class LabelsModule(nn.Module):
    nested: Dict[str, Dict[int, List[List[Labels]]]]

    def __init__(self, name):
        super().__init__()
        values = np.arange(2).reshape(-1, 1)
        self.labels = Labels([name], values)
        self.dict = {"labels": Labels([name], values)}
        self.list = [Labels([name], values)]
        self.tuple = tuple([Labels([name], values)])
        self.nested = {
            "dict": {42: [[Labels([name], values)], []], 50: []},
            "empty": {},
        }


class BlockModule(nn.Module):
    nested: Dict[str, Dict[int, List[List[TensorBlock]]]]

    def __init__(self, name):
        super().__init__()
        self.block = _create_block(name)
        self.dict = {"block": _create_block(name)}
        self.list = [_create_block(name)]
        self.tuple = tuple([_create_block(name)])
        self.nested = {
            "dict": {42: [[_create_block(name)], []], 50: []},
            "empty": {},
        }


class TensorModule(nn.Module):
    nested: Dict[str, Dict[int, List[List[TensorMap]]]]

    def __init__(self, name):
        super().__init__()
        self.tensor = _create_tensor(name)
        self.dict = {"tensor": _create_tensor(name)}
        self.list = [_create_tensor(name)]
        self.tuple = tuple([_create_tensor(name)])
        self.nested = {
            "dict": {42: [[_create_tensor(name)], []], 50: []},
            "empty": {},
        }


class EverythingModule(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.labels = Labels([name], np.arange(2).reshape(-1, 1))
        self.block = _create_block(name)
        self.tensor = _create_tensor(name)
        self.tuple = (
            Labels([name], np.arange(2).reshape(-1, 1)),
            _create_block(name),
            _create_tensor(name),
        )

        # check nested modules
        self.a = LabelsModule(name)
        self.b = BlockModule(name)
        self.c = TensorModule(name)


def test_to(devices_to_test):
    def check_device_dtype(module, device, dtype):
        assert module.labels.device == "cpu"

        assert module.block.device.type == device
        assert module.block.dtype == dtype

        assert module.tensor.device.type == device
        assert module.tensor.dtype == dtype

        assert module.tuple[0].device == "cpu"
        assert module.tuple[1].device.type == device
        assert module.tuple[1].dtype == dtype
        assert module.tuple[2].device.type == device
        assert module.tuple[2].dtype == dtype

        # labels are always on CPU for the metatensor-core backend
        assert module.a.labels.device == "cpu"
        assert module.a.dict["labels"].device == "cpu"
        assert module.a.list[0].device == "cpu"
        assert module.a.nested["dict"][42][0][0].device == "cpu"

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
    check_state_dict(state_dict["_extra_state"]["labels"])
    check_state_dict(state_dict["_extra_state"]["dict"]["labels"])
    check_state_dict(state_dict["_extra_state"]["list"][0])
    check_state_dict(state_dict["_extra_state"]["tuple"][0])
    check_state_dict(state_dict["_extra_state"]["nested"]["dict"][42][0][0])

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
