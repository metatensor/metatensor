import io
import os

import pytest
import torch
from packaging import version

import metatensor.torch


@pytest.fixture
def tensor():
    return metatensor.torch.load(
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
    unique_labels = metatensor.torch.unique_metadata(
        tensor,
        axis="samples",
        names=["system"],
    )

    # check type
    assert isinstance(unique_labels, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert unique_labels._type().name() == "Labels"

    # check label names
    assert unique_labels.names == ["system"]

    # repeat with gradients
    unique_labels = metatensor.torch.unique_metadata(
        tensor,
        axis="samples",
        names=["atom"],
        gradient="positions",
    )

    assert isinstance(unique_labels, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert unique_labels._type().name() == "Labels"

    assert unique_labels.names == ["atom"]


def test_unique_metadata_block(tensor):
    block = tensor.block(0)

    unique_labels = metatensor.torch.unique_metadata_block(
        block,
        axis="samples",
        names=["system"],
    )

    # check type
    assert isinstance(unique_labels, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert unique_labels._type().name() == "Labels"

    # check label names
    assert unique_labels.names == ["system"]

    # repeat with gradients
    unique_labels = metatensor.torch.unique_metadata_block(
        block,
        axis="samples",
        names=["atom"],
        gradient="positions",
    )

    assert isinstance(unique_labels, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert unique_labels._type().name() == "Labels"

    assert unique_labels.names == ["atom"]


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.unique_metadata, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
