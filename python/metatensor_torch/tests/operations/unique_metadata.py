import io
import os

import pytest
import torch

import metatensor.torch as mts


@pytest.fixture
def tensor():
    return mts.load(
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


def test_unique_metadata(tensor):
    unique_labels = mts.unique_metadata(
        tensor,
        axis="samples",
        names=["system"],
    )

    # check type
    assert isinstance(unique_labels, torch.ScriptObject)
    assert unique_labels._type().name() == "Labels"

    # check label names
    assert unique_labels.names == ["system"]

    # repeat with gradients
    unique_labels = mts.unique_metadata(
        tensor,
        axis="samples",
        names=["atom"],
        gradient="positions",
    )

    assert isinstance(unique_labels, torch.ScriptObject)
    assert unique_labels._type().name() == "Labels"

    assert unique_labels.names == ["atom"]


def test_unique_metadata_block(tensor):
    block = tensor.block(0)

    unique_labels = mts.unique_metadata_block(
        block,
        axis="samples",
        names=["system"],
    )

    # check type
    assert isinstance(unique_labels, torch.ScriptObject)
    assert unique_labels._type().name() == "Labels"

    # check label names
    assert unique_labels.names == ["system"]

    # repeat with gradients
    unique_labels = mts.unique_metadata_block(
        block,
        axis="samples",
        names=["atom"],
        gradient="positions",
    )

    assert isinstance(unique_labels, torch.ScriptObject)
    assert unique_labels._type().name() == "Labels"

    assert unique_labels.names == ["atom"]


@pytest.mark.skipif(os.environ.get("PYTORCH_JIT") == "0", reason="requires TorchScript")
def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(mts.unique_metadata, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
