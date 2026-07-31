import os

import numpy as np
import pytest

import metatensor as mts
from metatensor import Labels


torch = pytest.importorskip("torch")

from metatensor.learn.nn import InvariantLayerNorm, LayerNorm  # noqa: E402
from metatensor.learn.nn._layer_norm import _LayerNorm  # noqa: E402

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


def test_layer_norm_torch_mts_equivalence(tensor):
    """
    Tests that in the case of equivalent reduction dimensions that the native torch
    :py:class:`torch.nn.LayerNorm` and the re-implemented version in metatensor, i.e.
    :py:class:`_LayerNorm` are equivalent.

    This involves specifying the parameter `normlaized_shape` over the samples and
    components of each block, such that it is equivalent to specifying the `in_features`
    parameter in the metatensor version.
    """
    for block in tensor:
        # Apply the backend metatensor layer norm
        array = block.values
        norm_mts = _LayerNorm(
            in_features=array.shape[-1],  # i.e. properties
            elementwise_affine=True,
            dtype=torch.float64,
        )(array)

        # Apply the native torch.nn.LayerNorm. `normalized_shape` is defined over the
        # last D dimensions of the array, and we want to reduce over components and
        # properties
        norm_torch = torch.nn.LayerNorm(
            normalized_shape=array.shape[1:],
            elementwise_affine=True,
            dtype=torch.float64,
        )(array)
        assert torch.allclose(norm_mts, norm_torch, atol=1e-10, rtol=1e-10)


def test_equivariance(tensor, wigner_d_real):
    """
    Tests that application of an EquiLayerNorm layer is equivariant to O3
    transformation of the input.
    """
    # Define input and rotated input
    x = tensor
    Rx = wigner_d_real.transform_tensormap_o3(x)

    # Define the EquiLayerNorm module
    f = InvariantLayerNorm(
        in_keys=x.keys,
        in_features=[
            len(x.block(key).properties) for key in x.keys if key["o3_lambda"] == 0
        ],
        invariant_keys=Labels(
            ["o3_lambda"], np.array([0], dtype=np.int64).reshape(-1, 1)
        ),
        bias=True,  # should only bias the invariants
        dtype=torch.float64,
    )

    # Pass both through the linear layer
    Rfx = wigner_d_real.transform_tensormap_o3(f(x))  # R . f(x)
    fRx = f(Rx)  # f(R . x)

    assert mts.allclose(fRx, Rfx, atol=1e-10, rtol=1e-10)


def test_layer_norm_independent_samples():
    """
    Tests that the LayerNorm is applied to each sample independently.
    """
    layer_norm_module = _LayerNorm(
        in_features=10, elementwise_affine=True, dtype=torch.float64
    )
    tensor = torch.randn(5, 3, 10)
    norm = layer_norm_module(tensor)

    # Apply to each sample independently and check consistency
    for i in range(tensor.shape[0]):
        norm_i = layer_norm_module(tensor[i : i + 1])
        assert torch.allclose(norm[i], norm_i, atol=1e-10, rtol=1e-10)


def test_state_dict_structure():
    """
    Checks the exact structure (keys + metatensor tags) of ``LayerNorm`` and
    ``InvariantLayerNorm`` state_dicts to guard against regressions.
    """
    keys = Labels(
        names=["o3_lambda", "o3_sigma"],
        values=np.array([[0, 1], [1, 1]]),
    )
    invariant_keys = Labels(["o3_lambda"], np.array([0], dtype=np.int64).reshape(-1, 1))

    # --- LayerNorm with elementwise_affine=True, bias=True ---
    module = LayerNorm(
        in_keys=keys, in_features=[3, 3], elementwise_affine=True, bias=True
    )
    state_dict = module.state_dict()
    assert set(state_dict.keys()) == {
        "_mts_helper",
        "module_map._mts_helper",
        "module_map.module_list.0._mts_helper",
        "module_map.module_list.0.bias",
        "module_map.module_list.0.weight",
        "module_map.module_list.1._mts_helper",
        "module_map.module_list.1.bias",
        "module_map.module_list.1.weight",
        "_extra_state",
        "module_map._extra_state",
        "module_map.module_list.0._extra_state",
        "module_map.module_list.1._extra_state",
    }
    assert set(state_dict["_extra_state"].keys()) == set()
    assert set(state_dict["module_map._extra_state"].keys()) == {"_in_keys"}
    check_labels_state_dict(state_dict["module_map._extra_state"]["_in_keys"])
    assert set(state_dict["module_map.module_list.0._extra_state"].keys()) == set()
    assert set(state_dict["module_map.module_list.1._extra_state"].keys()) == set()

    # --- LayerNorm with elementwise_affine=False, bias=False ---
    module = LayerNorm(
        in_keys=keys, in_features=[3, 3], elementwise_affine=False, bias=False
    )
    state_dict = module.state_dict()
    assert set(state_dict.keys()) == {
        "_mts_helper",
        "module_map._mts_helper",
        "module_map.module_list.0._mts_helper",
        "module_map.module_list.1._mts_helper",
        "_extra_state",
        "module_map._extra_state",
        "module_map.module_list.0._extra_state",
        "module_map.module_list.1._extra_state",
    }
    assert set(state_dict["_extra_state"].keys()) == set()
    assert set(state_dict["module_map._extra_state"].keys()) == {"_in_keys"}
    check_labels_state_dict(state_dict["module_map._extra_state"]["_in_keys"])
    assert set(state_dict["module_map.module_list.0._extra_state"].keys()) == set()
    assert set(state_dict["module_map.module_list.1._extra_state"].keys()) == set()

    # --- InvariantLayerNorm: _LayerNorm on key 0, Identity on key 1 ---
    module = InvariantLayerNorm(
        in_keys=keys,
        in_features=[3],
        invariant_keys=invariant_keys,
        elementwise_affine=True,
        bias=True,
    )
    state_dict = module.state_dict()
    assert set(state_dict.keys()) == {
        "_mts_helper",
        "module_map._mts_helper",
        "module_map.module_list.0._mts_helper",
        "module_map.module_list.0.bias",
        "module_map.module_list.0.weight",
        "_extra_state",
        "module_map._extra_state",
        "module_map.module_list.0._extra_state",
    }
    assert set(state_dict["_extra_state"].keys()) == set()
    assert set(state_dict["module_map._extra_state"].keys()) == {"_in_keys"}
    check_labels_state_dict(state_dict["module_map._extra_state"]["_in_keys"])
    assert set(state_dict["module_map.module_list.0._extra_state"].keys()) == set()
