import os
from typing import Dict, List

import torch

from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.learn import nn


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

        self.register_buffer("labels", labels_value)
        self.register_buffer("nested", nested_value)
        self.dict = nn.Buffer(container_value)
        self.list = nn.Buffer(list_value)
        self.tuple = nn.Buffer(tuple_value)


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

        self.register_buffer("block", block_value)
        self.register_buffer("nested", nested_value)
        self.dict = nn.Buffer(container_value)
        self.list = nn.Buffer(list_value)
        self.tuple = nn.Buffer(tuple_value)


class TensorMapModule(nn.Module):
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

        self.register_buffer("tensor", tensor_value)
        self.register_buffer("nested", nested_value)
        self.dict = nn.Buffer(container_value)
        self.list = nn.Buffer(list_value)
        self.tuple = nn.Buffer(tuple_value)


class BufferModule(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.register_buffer(name, torch.rand(3, 4, dtype=torch.float64))


class ParameterModule(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.register_parameter(
            name, torch.nn.Parameter(torch.rand(3, 4, dtype=torch.float64))
        )


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

        # check nested modules
        self.a = LabelsModule(name)
        self.b = BlockModule(name)
        self.c = TensorMapModule(name)
        self.d = BufferModule(name)
        self.e = ParameterModule(name)


module = EverythingModule("test")
torch.jit.script(module).save(os.path.join(os.path.dirname(__file__), "test-module.pt"))
