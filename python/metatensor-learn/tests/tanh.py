import os

import pytest

import metatensor


torch = pytest.importorskip("torch")

from metatensor.learn.nn.tanh import InvariantTanh  # noqa: E402

from ._rotation_utils import WignerDReal  # noqa: E402


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
def wigner_d_real():
    return WignerDReal(lmax=4, angles=(0.87641, 1.8729, 0.9187))


def test_equivariance(tensor, wigner_d_real):
    """
    Tests that application of an invariant tanh layer is equivariant to O3
    transformation of the input.
    """
    # Define input and rotated input
    x = tensor
    Rx = wigner_d_real.transform_tensormap_o3(x)

    # Define the EquiLayerNorm module
    f = InvariantTanh(
        in_keys=x.keys,
        invariant_key_idxs=[i for i, key in enumerate(x.keys) if key["o3_lambda"] == 0],
    )

    # Pass both through the linear layer
    Rfx = wigner_d_real.transform_tensormap_o3(f(x))  # R . f(x)
    fRx = f(Rx)  # f(R . x)

    assert metatensor.allclose(fRx, Rfx, atol=1e-10, rtol=1e-10)
