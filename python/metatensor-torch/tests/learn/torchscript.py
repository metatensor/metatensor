"""
Module to test that metatensor.learn.nn modules 1) produce outputs that are invariant to
whether or not the module is jit scripted and 2) can be saved and loaded.
"""

import io

import pytest
import torch

import metatensor.torch
from metatensor.torch import allclose_raise
from metatensor.torch.learn.nn import EquivariantLinear, InvariantTanh, Linear, Tanh

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
        invariant_key_idxs=[0],
        in_features=len(tensor[0].properties),
        out_features=5,
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
    module = InvariantTanh(in_keys=tensor_no_grad.keys, invariant_key_idxs=[0])
    check_module_torch_script(module, tensor_no_grad)
    check_module_save_load(module)
