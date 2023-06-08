import torch

import equistore.torch


def test_classes():
    assert isinstance(equistore.torch.Labels, torch.ScriptClass)
    assert isinstance(equistore.torch.TensorBlock, torch.ScriptClass)
    assert isinstance(equistore.torch.TensorMap, torch.ScriptClass)
