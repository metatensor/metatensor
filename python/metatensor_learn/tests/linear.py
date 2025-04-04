import os

import numpy as np
import pytest

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


torch = pytest.importorskip("torch")

from metatensor.learn.nn import EquivariantLinear  # noqa: E402

from ._rotation_utils import WignerDReal  # noqa: E402
from ._tests_utils import random_single_block_no_components_tensor_map  # noqa: E402


DATA_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "metatensor_operations", "tests", "data"
)


@pytest.fixture
def tensor():
    tensor = metatensor.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.mts"))
    tensor = tensor.to(arrays="torch")
    tensor = metatensor.remove_gradients(tensor)
    return tensor


@pytest.fixture
def single_block_tensor():
    return random_single_block_no_components_tensor_map(use_torch=True)


@pytest.fixture
def equivariant_tensor():
    # Define a dummy invariant TensorBlock
    block_1 = TensorBlock(
        values=torch.randn(2, 1, 3, dtype=torch.float64),
        samples=Labels(
            ["system", "atom"],
            np.array(
                [
                    [0, 0],
                    [0, 1],
                ]
            ),
        ),
        components=[Labels(["o3_mu"], np.array([[0]]))],
        properties=Labels(["properties"], np.array([[0], [1], [2]])),
    )

    # Define a dummy covariant TensorBlock
    block_2 = TensorBlock(
        values=torch.randn(2, 3, 3, dtype=torch.float64),
        samples=Labels(
            ["system", "atom"],
            np.array(
                [
                    [0, 0],
                    [0, 1],
                ]
            ),
        ),
        components=[Labels(["o3_mu"], np.array([[-1], [0], [1]]))],
        properties=Labels(["properties"], np.array([[3], [4], [5]])),
    )

    # Create a TensorMap containing the dummy TensorBlocks
    keys = Labels(
        names=["o3_lambda", "o3_sigma"],
        values=np.array([[0, 1], [1, 1]]),
    )

    return TensorMap(keys, [block_1, block_2]).to(arrays="torch")


@pytest.mark.parametrize("bias", [True, False])
def test_equivariance(tensor, bias):
    """
    Tests that application of an EquivariantLinear layer is equivariant to O3
    transformation of the input.
    """
    wigner_d_real = WignerDReal(lmax=4, angles=(0.87641, 1.8729, 0.9187))

    # Define input and rotated input
    x = tensor
    Rx = wigner_d_real.transform_tensormap_o3(x)

    # Define the EquivariantLinear module
    f = EquivariantLinear(
        in_keys=x.keys,
        in_features=[len(x.block(key).properties) for key in x.keys],
        out_features=[len(x.block(key).properties) - 3 for key in x.keys],
        invariant_keys=metatensor.Labels(
            ["o3_lambda"], np.array([0], dtype=np.int64).reshape(-1, 1)
        ),
        bias=bias,  # this should only bias the invariant blocks
        dtype=torch.float64,
    )

    # Pass both through the linear layer
    Rfx = wigner_d_real.transform_tensormap_o3(f(x))  # R . f(x)
    fRx = f(Rx)  # f(R . x)

    assert metatensor.allclose(fRx, Rfx, atol=1e-10, rtol=1e-10)


def test_default_invariant_keys(equivariant_tensor):
    """
    Tests the default value of ``invariant_keys`` in EquivariantLinear.
    The module should be equivariant if applying a bias correctly to the
    default-identified invariant blocks.
    """
    wigner_d_real = WignerDReal(lmax=4, angles=(0.87641, 1.8729, 0.9187))

    # Define input and rotated input
    x = equivariant_tensor
    Rx = wigner_d_real.transform_tensormap_o3(x)

    f = EquivariantLinear(
        in_keys=x.keys,
        in_features=[len(x.block(key).properties) for key in x.keys],
        out_features=[len(x.block(key).properties) + 3 for key in x.keys],
        bias=True,  # this should only bias the invariant blocks
        dtype=torch.float64,
    )

    # Pass both through the linear layer
    Rfx = wigner_d_real.transform_tensormap_o3(f(x))  # R . f(x)
    fRx = f(Rx)  # f(R . x)

    assert metatensor.allclose(fRx, Rfx, atol=1e-10, rtol=1e-10)
