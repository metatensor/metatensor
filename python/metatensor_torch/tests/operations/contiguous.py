import io
import os

import pytest
import torch

import metatensor.torch as mts


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.is_contiguous, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(mts.make_contiguous, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
