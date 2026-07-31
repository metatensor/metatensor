import os

import numpy as np
import pytest

import metatensor as mts
from metatensor import Labels


torch = pytest.importorskip("torch")

from metatensor.learn import nn as nn  # noqa: E402

from ._rotation_utils import WignerDReal  # noqa: E402
from ._tests_utils import check_labels_state_dict  # noqa: E402


DATA_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "metatensor_operations", "tests", "data"
)


@pytest.fixture
def tensor():
    tensor = mts.load(os.path.join(DATA_ROOT, "qm7-spherical-expansion.mts"))
    tensor = tensor.to(arrays="torch")
    tensor = mts.remove_gradients(tensor)
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
            out_features=2,
            bias=True,
            dtype=torch.float64,
        ),
        nn.ReLU(in_keys=in_keys),
        nn.Linear(
            in_keys=in_keys,
            in_features=2,
            out_features=1,
            bias=True,
            dtype=torch.float64,
        ),
        nn.SiLU(in_keys=in_keys),
    )

    prediction = model(tensor)
    assert mts.equal_metadata(prediction, tensor, check=["samples", "components"])


def test_sequential_equi_mlp(tensor, wigner_d_real):
    """
    Constructs a small multi-layer perceptron from equivariant metatensor nn modules
    """
    in_keys = tensor.keys
    in_features = [len(tensor.block(key).properties) for key in in_keys]
    in_invariant_features = [
        len(tensor.block(key).properties) for key in in_keys if key["o3_lambda"] == 0
    ]
    invariant_keys = Labels(["o3_lambda"], np.array([0], dtype=np.int64).reshape(-1, 1))

    model = nn.Sequential(
        in_keys,
        nn.InvariantLayerNorm(
            in_keys=in_keys,
            in_features=in_invariant_features,
            invariant_keys=invariant_keys,
            dtype=torch.float64,
        ),
        nn.EquivariantLinear(
            in_keys=in_keys,
            in_features=in_features,
            out_features=4,
            invariant_keys=invariant_keys,
            bias=True,
            dtype=torch.float64,
        ),
        nn.InvariantTanh(in_keys=in_keys, invariant_keys=invariant_keys),
        nn.EquivariantLinear(
            in_keys=in_keys,
            in_features=4,
            out_features=2,
            invariant_keys=invariant_keys,
            bias=True,
            dtype=torch.float64,
        ),
        nn.InvariantReLU(in_keys=in_keys, invariant_keys=invariant_keys),
        nn.EquivariantLinear(
            in_keys=in_keys,
            in_features=2,
            out_features=1,
            invariant_keys=invariant_keys,
            bias=True,
            dtype=torch.float64,
        ),
        nn.InvariantSiLU(in_keys=in_keys, invariant_keys=invariant_keys),
    )

    prediction = model(tensor)
    assert mts.equal_metadata(prediction, tensor, check=["samples", "components"])

    # Test equivariance
    # Define input and rotated input
    Rx = wigner_d_real.transform_tensormap_o3(tensor)
    fRx = model(Rx)
    Rfx = wigner_d_real.transform_tensormap_o3(prediction)

    assert mts.allclose(fRx, Rfx, atol=1e-10, rtol=1e-10)


def test_state_dict_structure():
    """
    Checks the exact structure (keys + metatensor tags) of ``Sequential``
    state_dict to guard against regressions.
    """
    keys = Labels(
        names=["o3_lambda", "o3_sigma"],
        values=np.array([[0, 1], [1, 1]]),
    )

    model = nn.Sequential(
        keys,
        nn.Linear(in_keys=keys, in_features=[3, 3], out_features=2, bias=True),
        nn.Tanh(in_keys=keys),
        nn.Linear(in_keys=keys, in_features=[2, 2], out_features=1, bias=True),
    )
    state_dict = model.state_dict()
    assert set(state_dict.keys()) == {
        "_mts_helper",
        "module_map._mts_helper",
        "module_map.module_list.0.0.bias",
        "module_map.module_list.0.0.weight",
        "module_map.module_list.0.2.bias",
        "module_map.module_list.0.2.weight",
        "module_map.module_list.1.0.bias",
        "module_map.module_list.1.0.weight",
        "module_map.module_list.1.2.bias",
        "module_map.module_list.1.2.weight",
        "_extra_state",
        "module_map._extra_state",
    }
    assert set(state_dict["_extra_state"].keys()) == set()
    assert set(state_dict["module_map._extra_state"].keys()) == {"_in_keys"}
    check_labels_state_dict(state_dict["module_map._extra_state"]["_in_keys"])
