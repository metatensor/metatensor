import torch

from metatensor.torch import Labels, TensorBlock, TensorMap


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


class LabelsModule(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        values = torch.arange(2).reshape(-1, 1)
        self.labels = Labels([name], values)
        self.dict = {"labels": Labels([name], values)}
        self.list = [Labels([name], values)]
        self.tuple = tuple([Labels([name], values)])
        self.nested = {"dict": {42: [Labels([name], values)]}}


class BlockModule(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.block = _create_block(name)
        self.dict = {"block": _create_block(name)}
        self.list = [_create_block(name)]
        self.tuple = tuple([_create_block(name)])
        self.nested = {"dict": {42: [_create_block(name)]}}


class TensorModule(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.tensor = _create_tensor(name)
        self.dict = {"tensor": _create_tensor(name)}
        self.list = [_create_tensor(name)]
        self.tuple = tuple([_create_tensor(name)])
        self.nested = {"dict": {42: [_create_tensor(name)]}}


class EverythingModule(torch.nn.Module):
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


module = EverythingModule("test")
torch.jit.script(module).save("test-module.pt")
