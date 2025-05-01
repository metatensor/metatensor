import io
import os

import pytest
import torch

import metatensor.torch as mts
from metatensor.torch import Labels


def test_one_hot():
    original_labels = Labels(
        names=["atom", "type"],
        values=torch.tensor([[0, 6], [1, 1], [4, 6]]),
    )
    possible_labels = Labels("type", torch.tensor([[1], [6]]))
    correct_encoding = torch.tensor([[0, 1], [1, 0], [0, 1]])
    one_hot_encoding = mts.one_hot(original_labels, possible_labels)

    assert isinstance(one_hot_encoding, torch.Tensor)
    assert torch.all(one_hot_encoding == correct_encoding)


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.one_hot, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
