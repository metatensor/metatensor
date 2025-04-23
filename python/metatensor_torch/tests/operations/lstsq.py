import io
import os

import pytest
import torch

import metatensor.torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def test_lstsq():
    X_tensor = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1, 0], [0, 1]], dtype=torch.float64),
                samples=Labels.range("s", 2),
                components=[],
                properties=Labels.range("p1", 2),
            )
        ],
    )
    Y_tensor = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1, 0], [0, 1]], dtype=torch.float64),
                samples=Labels.range("s", 2),
                components=[],
                properties=Labels.range("p2", 2),
            )
        ],
    )
    solution_tensor = metatensor.torch.lstsq(X_tensor, Y_tensor, rcond=1e-14)

    # check output type
    assert isinstance(solution_tensor, torch.ScriptObject)
    assert solution_tensor._type().name() == "TensorMap"

    # check content
    expected_solution = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1, 0], [0, 1]], dtype=torch.float64),
                samples=Labels.range("p2", 2),
                components=[],
                properties=Labels.range("p1", 2),
            )
        ],
    )
    assert metatensor.torch.equal(solution_tensor, expected_solution)


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.lstsq, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
