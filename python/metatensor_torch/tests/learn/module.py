import pytest
import torch

from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.learn import nn

from . import _tests_utils


DEVICES_TO_TEST = []
if _tests_utils.can_use_mps_backend():
    DEVICES_TO_TEST.append("mps")
if torch.cuda.is_available():
    DEVICES_TO_TEST.append("cuda")


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
    def __init__(self, name):
        super().__init__()
        values = torch.arange(2).reshape(-1, 1)
        self.labels = Labels([name], values)
        self.dict = {"labels": Labels([name], values)}
        self.list = [Labels([name], values)]
        self.tuple = tuple([Labels([name], values)])
        self.nested = {"dict": {42: [Labels([name], values)]}}

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
            return self.nested["dict"][42][0]
        else:
            return self.labels


class BlockModule(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.block = _create_block(name)
        self.dict = {"block": _create_block(name)}
        self.list = [_create_block(name)]
        self.tuple = tuple([_create_block(name)])
        self.nested = {"dict": {42: [_create_block(name)]}}

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
            return self.nested["dict"][42][0]
        else:
            return self.block


class TensorModule(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.tensor = _create_tensor(name)
        self.dict = {"tensor": _create_tensor(name)}
        self.list = [_create_tensor(name)]
        self.tuple = tuple([_create_tensor(name)])
        self.nested = {"dict": {42: [_create_tensor(name)]}}

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
            return self.nested["dict"][42][0]
        else:
            return self.tensor


class EverythingModule(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.labels = Labels([name], torch.arange(2).reshape(-1, 1))
        self.block = _create_block(name)
        self.tensor = _create_tensor(name)
        self.tuple = (
            Labels([name], torch.arange(2).reshape(-1, 1)),
            _create_block(name),
            _create_tensor(name),
        )

        # check nested modules
        self.a = LabelsModule(name)
        self.b = BlockModule(name)
        self.c = TensorModule(name)


@pytest.mark.parametrize("scripted", [True, False])
def test_to(scripted):
    def check_device_dtype(module, device, dtype):
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
        assert module.a.nested["dict"][42][0].device.type == device

        assert module.b.block.device.type == device
        assert module.b.block.dtype == dtype
        assert module.b.dict["block"].device.type == device
        assert module.b.dict["block"].dtype == dtype
        assert module.b.list[0].device.type == device
        assert module.b.list[0].dtype == dtype
        assert module.b.tuple[0].device.type == device
        assert module.b.tuple[0].dtype == dtype
        assert module.b.nested["dict"][42][0].device.type == device
        assert module.b.nested["dict"][42][0].dtype == dtype

        assert module.c.tensor.device.type == device
        assert module.c.tensor.dtype == dtype
        assert module.c.dict["tensor"].device.type == device
        assert module.c.dict["tensor"].dtype == dtype
        assert module.c.list[0].device.type == device
        assert module.c.list[0].dtype == dtype
        assert module.c.tuple[0].device.type == device
        assert module.c.tuple[0].dtype == dtype
        assert module.c.nested["dict"][42][0].device.type == device
        assert module.c.nested["dict"][42][0].dtype == dtype

    module = EverythingModule("test")
    if scripted:
        module = torch.jit.script(module)

    check_device_dtype(module, "cpu", dtype_to_int(torch.float64))

    module = module.to(dtype=torch.float32)
    check_device_dtype(module, "cpu", dtype_to_int(torch.float32))

    for device in DEVICES_TO_TEST:
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

    for device in DEVICES_TO_TEST:
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
