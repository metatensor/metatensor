import os

import numpy as np
import pytest

import metatensor


torch = pytest.importorskip("torch")

from metatensor.learn.nn import EquivariantLinear  # noqa: E402

from ._rotation_utils import WignerDReal  # noqa: E402
from ._tests_utils import random_single_block_no_components_tensor_map  # noqa: E402


DATA_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "metatensor-operations", "tests", "data"
)


@pytest.fixture
def tensor():
    tensor = metatensor.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"))
    tensor = tensor.to(arrays="torch")
    tensor = metatensor.remove_gradients(tensor)
    return tensor


@pytest.fixture
def single_block_tensor():
    return random_single_block_no_components_tensor_map(use_torch=True)


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
