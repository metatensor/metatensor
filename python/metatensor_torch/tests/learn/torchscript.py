"""
Module to test that metatensor.learn.nn modules 1) produce outputs that are invariant to
whether or not the module is jit scripted and 2) can be saved and loaded.
"""

import io

import pytest
import torch

import metatensor.torch
from metatensor.torch import allclose_raise
from metatensor.torch.learn.nn import (
    EquivariantLinear,
    EquivariantTransformation,
    InvariantLayerNorm,
    InvariantReLU,
    InvariantSiLU,
    InvariantTanh,
    LayerNorm,
    Linear,
    ReLU,
    Sequential,
    SiLU,
    Tanh,
)

from ._tests_utils import random_single_block_no_components_tensor_map


@pytest.fixture
def tensor():
    """Returns a single block tensor map"""
    return random_single_block_no_components_tensor_map().to(torch.float64)


@pytest.fixture
def tensor_no_grad():
    """Returns a single block tensor map with no gradients"""
    return metatensor.torch.remove_gradients(
        random_single_block_no_components_tensor_map().to(torch.float64)
    )


def check_module_torch_script(module, tensor):
    """Tests output of module is invariant to torchscripting"""
    ref_tensor = module(tensor)  # apply module
    module_scripted = torch.jit.script(module)  # script
    out_tensor = module_scripted(tensor)  # apply scripted
    allclose_raise(ref_tensor, out_tensor)  # check same


def check_module_save_load(module):
    """Tests save and load of module"""
    module_scripted = torch.jit.script(module)  # script
    with io.BytesIO() as buffer:
        torch.jit.save(module_scripted, buffer)  # save
        buffer.seek(0)
        torch.jit.load(buffer)  # load


def test_linear(tensor):
    """Tests module Linear"""
    module = Linear(
        in_keys=tensor.keys,
        in_features=len(tensor[0].properties),
        bias=True,
        out_features=5,
        dtype=torch.float64,
    )
    check_module_torch_script(module, tensor)
    check_module_save_load(module)


def test_equivariant_linear(tensor):
    """Tests module EquivariantLinear"""
    module = EquivariantLinear(
        in_keys=tensor.keys,
        in_features=len(tensor[0].properties),
        out_features=5,
        invariant_keys=metatensor.torch.Labels(
            ["_"],
            torch.tensor([0], dtype=torch.int64).reshape(-1, 1),
        ),
        dtype=torch.float64,
    )
    check_module_torch_script(module, tensor)
    check_module_save_load(module)


def test_tanh(tensor_no_grad):
    """Tests module Tanh"""
    module = Tanh(in_keys=tensor_no_grad.keys)
    check_module_torch_script(module, tensor_no_grad)
    check_module_save_load(module)


def test_invariant_tanh(tensor_no_grad):
    """Tests module InvariantTanh"""
    module = InvariantTanh(
        in_keys=tensor_no_grad.keys,
        invariant_keys=metatensor.torch.Labels(
            ["_"],
            torch.tensor([0], dtype=torch.int64).reshape(-1, 1),
        ),
    )
    check_module_torch_script(module, tensor_no_grad)
    check_module_save_load(module)


def test_relu(tensor_no_grad):
    """Tests module ReLU"""
    module = ReLU(in_keys=tensor_no_grad.keys)
    check_module_torch_script(module, tensor_no_grad)
    check_module_save_load(module)


def test_invariant_relu(tensor_no_grad):
    """Tests module InvariantReLU"""
    module = InvariantReLU(
        in_keys=tensor_no_grad.keys,
        invariant_keys=metatensor.torch.Labels(
            ["_"],
            torch.tensor([0], dtype=torch.int64).reshape(-1, 1),
        ),
    )
    check_module_torch_script(module, tensor_no_grad)
    check_module_save_load(module)


def test_silu(tensor_no_grad):
    """Tests module SiLU"""
    module = SiLU(in_keys=tensor_no_grad.keys)
    check_module_torch_script(module, tensor_no_grad)
    check_module_save_load(module)


def test_invariant_silu(tensor_no_grad):
    """Tests module InvariantSiLU"""
    module = InvariantSiLU(
        in_keys=tensor_no_grad.keys,
        invariant_keys=metatensor.torch.Labels(
            ["_"],
            torch.tensor([0], dtype=torch.int64).reshape(-1, 1),
        ),
    )
    check_module_torch_script(module, tensor_no_grad)
    check_module_save_load(module)


def test_layer_norm(tensor_no_grad):
    """Tests module LayerNorm"""
    module = LayerNorm(
        in_keys=tensor_no_grad.keys,
        in_features=len(tensor_no_grad[0].properties),
        dtype=torch.float64,
    )
    check_module_torch_script(module, tensor_no_grad)
    check_module_save_load(module)


def test_invariant_layer_norm(tensor_no_grad):
    """Tests module InvariantLayerNorm"""
    module = InvariantLayerNorm(
        in_keys=tensor_no_grad.keys,
        in_features=len(tensor_no_grad[0].properties),
        invariant_keys=metatensor.torch.Labels(
            ["_"],
            torch.tensor([0], dtype=torch.int64).reshape(-1, 1),
        ),
        dtype=torch.float64,
    )
    check_module_torch_script(module, tensor_no_grad)
    check_module_save_load(module)


def test_sequential(tensor):
    """Tests module Sequential"""
    in_keys = tensor.keys
    in_features = [len(tensor.block(key).properties) for key in in_keys]
    module = Sequential(
        in_keys,
        Linear(
            in_keys=in_keys,
            in_features=in_features,
            out_features=4,
            bias=True,
            dtype=torch.float64,
        ),
        Tanh(in_keys=in_keys),
    )
    check_module_torch_script(module, tensor)
    check_module_save_load(module)


def test_equivariant_transform(tensor):
    """Tests module EquivariantTransformation"""

    def module_wrapper(in_features, device, dtype):
        """
        A sequential module with nonlinearities
        """
        return torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.Linear(
                in_features=in_features,
                out_features=5,
                device=device,
                dtype=dtype,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=5,
                out_features=in_features,
                device=device,
                dtype=dtype,
            ),
        )

    in_keys = tensor.keys
    in_features = [len(tensor.block(key).properties) for key in in_keys]

    modules = [
        module_wrapper(
            in_feat, device=tensor.device, dtype=tensor.block(0).values.dtype
        )
        for in_feat in in_features
    ]

    module = EquivariantTransformation(
        modules,
        in_keys,
        in_features,
        out_properties=[tensor.block(key).properties for key in tensor.keys],
        invariant_keys=in_keys,
    )
    check_module_torch_script(module, tensor)
    check_module_save_load(module)
