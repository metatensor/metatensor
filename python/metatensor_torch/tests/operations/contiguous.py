import io
import os

import pytest
import torch

import metatensor.torch


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.is_contiguous, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.make_contiguous, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
