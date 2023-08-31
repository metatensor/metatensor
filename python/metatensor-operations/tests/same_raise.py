"""Check that errors are raised for unequal meta data in several operations."""

from os import path

import numpy as np
import pytest

import metatensor
from metatensor import NotEqualError


DATA_ROOT = path.join(path.dirname(__file__), "data")


def tensor():
    tensor = metatensor.load(
        path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        use_numpy=True,
    )

    msg = (
        "Tensor must have at least one gradient. When no gradients are present certain "
        "tests will pass without testing anything."
    )
    assert len(tensor.block(0).gradients_list()) > 0, msg

    return tensor


@pytest.mark.parametrize(
    "operation_str",
    ["add", "divide", "dot", "join", "lstsq", "multiply", "solve", "subtract"],
)
def test_different_keys(operation_str):
    operation = getattr(metatensor, operation_str)

    A = tensor()
    keys = metatensor.Labels(names="foo", values=np.array([[0]]))
    B = metatensor.TensorMap(keys, [A[0].copy()])

    with pytest.raises(NotEqualError, match="should have the same keys"):
        if operation_str == "join":
            operation([A, B], axis="properties")
        elif operation_str == "lstsq":
            operation(A, B, rcond=1)
        else:
            operation(A, B)


@pytest.mark.parametrize(
    "operation_str", ["add", "divide", "lstsq", "multiply", "subtract"]
)
def test_different_gradients(operation_str):
    operation = getattr(metatensor, operation_str)

    A = tensor()
    B = metatensor.remove_gradients(tensor())

    with pytest.raises(NotEqualError, match="should have the same gradient parameters"):
        if operation_str == "lstsq":
            operation(A, B, rcond=1)
        else:
            operation(A, B)


@pytest.mark.parametrize("operation_str", ["add", "divide", "multiply", "subtract"])
def test_different_blocks(operation_str):
    operation = getattr(metatensor, operation_str)

    values = np.array([[0]])
    samples = metatensor.Labels.single()

    properties_1 = metatensor.Labels(names="props1", values=np.array([[0]]))
    properties_2 = metatensor.Labels(names="props2", values=np.array([[0]]))

    block_1 = metatensor.TensorBlock(
        values=values, samples=samples, components=[], properties=properties_1
    )
    block_2 = metatensor.TensorBlock(
        values=values, samples=samples, components=[], properties=properties_2
    )

    keys = metatensor.Labels(names="foo", values=np.array([[0]]))

    A = metatensor.TensorMap(keys, [block_1])
    B = metatensor.TensorMap(keys, [block_2])

    with pytest.raises(NotEqualError, match="should have the same properties"):
        operation(A, B)
