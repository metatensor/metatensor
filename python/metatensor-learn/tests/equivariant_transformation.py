import os

import numpy as np
import pytest

import metatensor


torch = pytest.importorskip("torch")

from metatensor.learn.nn.equivariant_transformation import (  # noqa: E402
    EquivariantTransformation,  # noqa: E402
)  # noqa:E402

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


def module_wrapper(in_features, device, dtype):
    """
    A sequential module with nonlinearities
    """
    return torch.nn.Sequential(
        torch.nn.Tanh(),
        torch.nn.Linear(
            in_features=in_features,
            out_features=5,
            device=device,
            dtype=dtype,
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(
            in_features=5,
            out_features=in_features,
            device=device,
            dtype=dtype,
        ),
    )


def test_equivariance(tensor, wigner_d_real):
    """
    Tests that application of the EquivariantTransformation layer is equivariant to O3
    transformation of the input.
    """
    # Define input and rotated input
    x = tensor
    Rx = wigner_d_real.transform_tensormap_o3(x)

    in_features = [len(x.block(key).properties) for key in x.keys]
    modules = [
        module_wrapper(in_feat, device=x.device, dtype=x.block(0).values.dtype)
        for in_feat in in_features
    ]

    # Define the EquiLayerNorm module
    f = EquivariantTransformation(
        modules,
        x.keys,
        in_features,
        out_properties=[x.block(key).properties for key in x.keys],
        invariant_keys=metatensor.Labels(
            ["o3_lambda"], np.array([0], dtype=np.int64).reshape(-1, 1)
        ),
    )

    # Pass both through the linear layer
    Rfx = wigner_d_real.transform_tensormap_o3(f(x))  # R . f(x)
    fRx = f(Rx)  # f(R . x)

    assert metatensor.allclose(fRx, Rfx, atol=1e-10, rtol=1e-10)
