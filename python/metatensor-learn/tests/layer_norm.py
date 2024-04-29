import pytest

import metatensor


torch = pytest.importorskip("torch")

from metatensor.learn.nn.layer_norm import InvariantLayerNorm, _LayerNorm  # noqa: E402

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

        # Apply the native torch.nn.LayerNorm. This requires reshaping such that the
        # properties are in the first dimension. This is because the `normalized_shape`
        # parameter is defined over the *final* D dimensions of the array, and we want
        # to reduce over samples and components.
        array_T = array.transpose(0, 2)
        norm_torch = torch.nn.LayerNorm(
            normalized_shape=array_T.shape[1:],
            elementwise_affine=True,
            dtype=torch.float64,
        )(array_T).transpose(0, 2)
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
        invariant_key_idxs=[i for i, key in enumerate(x.keys) if key["o3_lambda"] == 0],
        in_features=[
            len(x.block(key).properties) for key in x.keys if key["o3_lambda"] == 0
        ],
        bias=True,  # should only bias the invariants
        dtype=torch.float64,
    )

    # Pass both through the linear layer
    Rfx = wigner_d_real.transform_tensormap_o3(f(x))  # R . f(x)
    fRx = f(Rx)  # f(R . x)

    assert metatensor.allclose(fRx, Rfx, atol=1e-10, rtol=1e-10)
