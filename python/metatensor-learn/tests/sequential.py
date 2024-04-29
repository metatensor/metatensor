import pytest

import metatensor


torch = pytest.importorskip("torch")

from metatensor.learn import nn as nn  # noqa: E402

from ._rotation_utils import WignerDReal  # noqa: E402


@pytest.fixture
def tensor():
    tensor = metatensor.load(
        "../metatensor-operations/tests/data/qm7-spherical-expansion.npz",
        use_numpy=True,
    ).to(arrays="torch")
    tensor = metatensor.remove_gradients(tensor)
    return tensor


@pytest.fixture
def wigner_d_real():
    return WignerDReal(lmax=4, angles=(0.87641, 1.8729, 0.9187))


def test_sequential_mlp(tensor):
    """
    Constructs a small multi-layer perceptron from standard metatensor nn modules
    """
    in_keys = tensor.keys
    in_features = [len(tensor.block(key).properties) for key in in_keys]

    model = nn.Sequential(
        in_keys,
        nn.LayerNorm(
            in_keys=in_keys,
            in_features=in_features,
            dtype=torch.float64,
        ),
        nn.Linear(
            in_keys=in_keys,
            in_features=in_features,
            out_features=4,
            bias=True,
            dtype=torch.float64,
        ),
        nn.Tanh(in_keys=in_keys),
        nn.Linear(
            in_keys=in_keys,
            in_features=4,
            out_features=1,
            bias=True,
            dtype=torch.float64,
        ),
    )

    prediction = model(tensor)
    assert metatensor.equal_metadata(
        prediction, tensor, check=["samples", "components"]
    )


def test_sequential_equi_mlp(tensor, wigner_d_real):
    """
    Constructs a small multi-layer perceptron from equivariant metatensor nn modules
    """
    in_keys = tensor.keys
    in_features = [len(tensor.block(key).properties) for key in in_keys]
    in_invariant_features = [
        len(tensor.block(key).properties) for key in in_keys if key["o3_lambda"] == 0
    ]
    invariant_key_idxs = [i for i, key in enumerate(in_keys) if key["o3_lambda"] == 0]

    model = nn.Sequential(
        in_keys,
        nn.InvariantLayerNorm(
            in_keys=in_keys,
            invariant_key_idxs=invariant_key_idxs,
            in_features=in_invariant_features,
            dtype=torch.float64,
        ),
        nn.EquivariantLinear(
            in_keys=in_keys,
            invariant_key_idxs=invariant_key_idxs,
            in_features=in_features,
            out_features=4,
            bias=True,
            dtype=torch.float64,
        ),
        nn.InvariantTanh(in_keys=in_keys, invariant_key_idxs=invariant_key_idxs),
        nn.EquivariantLinear(
            in_keys=in_keys,
            invariant_key_idxs=invariant_key_idxs,
            in_features=4,
            out_features=1,
            bias=True,
            dtype=torch.float64,
        ),
    )

    prediction = model(tensor)
    assert metatensor.equal_metadata(
        prediction, tensor, check=["samples", "components"]
    )

    # Test equivariance
    # Define input and rotated input
    Rx = wigner_d_real.transform_tensormap_o3(tensor)
    fRx = model(Rx)
    Rfx = wigner_d_real.transform_tensormap_o3(prediction)

    assert metatensor.allclose(fRx, Rfx, atol=1e-10, rtol=1e-10)
