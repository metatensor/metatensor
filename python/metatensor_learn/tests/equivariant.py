import os

import numpy as np
import pytest

import metatensor as mts
from metatensor import Labels


torch = pytest.importorskip("torch")

from metatensor.learn.nn import EquivariantTransformation  # noqa: E402

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
        invariant_keys=Labels(
            ["o3_lambda"], np.array([0], dtype=np.int64).reshape(-1, 1)
        ),
    )

    # Pass both through the linear layer
    Rfx = wigner_d_real.transform_tensormap_o3(f(x))  # R . f(x)
    fRx = f(Rx)  # f(R . x)

    assert mts.allclose(fRx, Rfx, atol=1e-10, rtol=1e-10)


def test_state_dict_structure():
    """
    Checks the exact structure (keys + metatensor tags) of
    ``EquivariantTransformation`` state_dict to guard against regressions.
    """
    keys = Labels(
        names=["o3_lambda", "o3_sigma"],
        values=np.array([[0, 1], [1, 1]]),
    )
    invariant_keys = Labels(["o3_lambda"], np.array([0], dtype=np.int64).reshape(-1, 1))
    out_props = [
        Labels(["p"], np.array([[0], [1]])),
        Labels(["p"], np.array([[0], [1]])),
    ]

    # --- with out_properties ---
    module = EquivariantTransformation(
        modules=[torch.nn.Tanh(), torch.nn.Tanh()],
        in_keys=keys,
        in_features=[3, 3],
        out_properties=out_props,
        invariant_keys=invariant_keys,
    )
    state_dict = module.state_dict()
    assert set(state_dict.keys()) == {
        "_mts_helper",
        "module_map._mts_helper",
        "module_map.module_list.1._mts_helper",
        "_extra_state",
        "module_map._extra_state",
        "module_map.module_list.1._extra_state",
    }
    assert set(state_dict["_extra_state"].keys()) == set()
    assert set(state_dict["module_map._extra_state"].keys()) == {
        "_in_keys",
        "_out_properties",
    }
    check_labels_state_dict(state_dict["module_map._extra_state"]["_in_keys"])
    out_properties = state_dict["module_map._extra_state"]["_out_properties"]
    assert isinstance(out_properties, list)
    assert len(out_properties) == 2
    for entry in out_properties:
        check_labels_state_dict(entry)
    assert set(state_dict["module_map.module_list.1._extra_state"].keys()) == set()

    # --- without out_properties ---
    module = EquivariantTransformation(
        modules=[torch.nn.Tanh(), torch.nn.Tanh()],
        in_keys=keys,
        in_features=[3, 3],
        out_features=[3, 3],
        invariant_keys=invariant_keys,
    )
    state_dict = module.state_dict()
    assert set(state_dict.keys()) == {
        "_mts_helper",
        "module_map._mts_helper",
        "module_map.module_list.1._mts_helper",
        "_extra_state",
        "module_map._extra_state",
        "module_map.module_list.1._extra_state",
    }
    assert set(state_dict["_extra_state"].keys()) == set()
    assert set(state_dict["module_map._extra_state"].keys()) == {"_in_keys"}
    check_labels_state_dict(state_dict["module_map._extra_state"]["_in_keys"])
    assert set(state_dict["module_map.module_list.1._extra_state"].keys()) == set()
