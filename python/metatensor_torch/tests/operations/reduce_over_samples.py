import io
import os
import sys

import pytest
import torch
from packaging import version

import metatensor.torch
from metatensor.torch import Labels, TensorBlock, TensorMap


TORCH_VERSION = version.parse(torch.__version__)


def check_operation(reduce_over_samples):
    tensor = metatensor.torch.load(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "metatensor_operations",
            "tests",
            "data",
            "qm7-power-spectrum.mts",
        )
    )

    assert tensor.sample_names == ["system", "atom"]
    reduced_tensor = reduce_over_samples(tensor, "atom")

    assert isinstance(reduced_tensor, torch.ScriptObject)
    assert reduced_tensor.sample_names == ["system"]


@pytest.mark.skipif(
    TORCH_VERSION.major == 2
    and TORCH_VERSION.minor == 4
    and sys.platform.startswith("win32"),
    reason=(
        "These tests cause a memory corruption in scatter_add, "
        "but only when running from pytest, with torch 2.4 on Windows"
    ),
)
def test_reduce_over_samples():
    check_operation(metatensor.torch.sum_over_samples)
    check_operation(metatensor.torch.mean_over_samples)
    check_operation(metatensor.torch.std_over_samples)
    check_operation(metatensor.torch.var_over_samples)


def test_reduction_all_samples():
    """Check that reducing all samples to a single one works"""
    block_1 = TensorBlock(
        values=torch.tensor(
            [
                [1, 2, 4],
                [3, 5, 6],
                [-1.3, 26.7, 4.54],
            ]
        ),
        samples=Labels.range("s", 3),
        components=[],
        properties=Labels(["p"], torch.tensor([[0], [1], [5]])),
    )
    keys = Labels(names=["key_1", "key_2"], values=torch.tensor([[0, 0]]))
    X = TensorMap(keys, [block_1])

    sum_X = metatensor.torch.sum_over_samples(X, sample_names=["s"])
    mean_X = metatensor.torch.mean_over_samples(X, sample_names=["s"])
    var_X = metatensor.torch.var_over_samples(X, sample_names=["s"])
    std_X = metatensor.torch.std_over_samples(X, sample_names=["s"])

    assert sum_X[0].samples == Labels.single()
    assert metatensor.torch.equal_metadata(sum_X, mean_X)
    assert metatensor.torch.equal_metadata(sum_X, std_X)
    assert metatensor.torch.equal_metadata(mean_X, var_X)

    assert torch.all(sum_X[0].values == torch.sum(X[0].values, axis=0))
    assert torch.all(mean_X[0].values == torch.mean(X[0].values, axis=0))
    assert torch.allclose(std_X[0].values, torch.std(X[0].values, correction=0, axis=0))
    assert torch.allclose(var_X[0].values, torch.var(X[0].values, correction=0, axis=0))


def test_save():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.sum_over_samples, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.mean_over_samples, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.std_over_samples, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.var_over_samples, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
