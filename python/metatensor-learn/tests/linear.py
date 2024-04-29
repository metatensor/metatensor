import pytest

import metatensor


torch = pytest.importorskip("torch")

from metatensor.learn.nn import EquivariantLinear, Linear  # noqa: E402

from ._rotation_utils import WignerDReal  # noqa: E402
from ._tests_utils import random_single_block_no_components_tensor_map  # noqa: E402


@pytest.fixture()
def single_block_tensor_torch():
    """
    random tensor map with no components using torch as array backend
    """
    return random_single_block_no_components_tensor_map(use_torch=True)


@pytest.fixture(scope="module", autouse=True)
def set_random_generator():
    """Set the random generator to same seed before each test is run.
    Otherwise test behaviour is dependend on the order of the tests
    in this file and the number of parameters of the test.
    """
    torch.random.manual_seed(122578741812)


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


def test_linear_single_block_tensor(single_block_tensor_torch):
    # testing initialization by non sequence arguments
    tensor_module_init_nonseq = Linear(
        in_keys=single_block_tensor_torch.keys,
        in_features=[2],
        out_features=[2],
        bias=[True],
        out_properties=[single_block_tensor_torch[0].properties],
    )
    # testing initialization by sequence arguments
    tensor_module_init_seq = Linear(
        in_keys=single_block_tensor_torch.keys,
        in_features=2,
        out_features=2,
        bias=True,
        out_properties=single_block_tensor_torch[0].properties,
    )
    for i in range(len(tensor_module_init_seq.module_map)):
        assert (
            tensor_module_init_seq.module_map[i].in_features
            == tensor_module_init_nonseq.module_map[i].in_features
        ), (
            "in_features differ when using sequential and non sequential input for"
            " initialization"
        )
        assert (
            tensor_module_init_seq.module_map[i].out_features
            == tensor_module_init_nonseq.module_map[i].out_features
        ), (
            "out_features differ when using sequential and non sequential input for"
            " initialization"
        )
        assert (
            tensor_module_init_seq.module_map[i].bias.shape
            == tensor_module_init_nonseq.module_map[i].bias.shape
        ), (
            "bias differ when using sequential and non sequential input for"
            " initialization"
        )

    tensor_module = tensor_module_init_nonseq

    with torch.no_grad():
        out_tensor = tensor_module(single_block_tensor_torch)

    for i, item in enumerate(single_block_tensor_torch.items()):
        key, block = item
        module = tensor_module.module_map[i]
        assert (
            tensor_module.module_map.get_module(key) is module
        ), "modules should be initialized in the same order as keys"

        with torch.no_grad():
            ref_values = module(block.values)
        out_block = out_tensor.block(key)
        assert torch.allclose(ref_values, out_block.values)
        assert block.properties == out_block.properties

        for parameter, gradient in block.gradients():
            with torch.no_grad():
                ref_gradient_values = module(gradient.values)
            out_gradient = out_block.gradient(parameter)
            assert torch.allclose(ref_gradient_values, out_gradient.values)
            assert gradient.properties == out_gradient.properties


@pytest.mark.parametrize("bias", [True, False])
def test_equivariance(tensor, wigner_d_real, bias):
    """
    Tests that application of an EquivariantLinear layer is equivariant to O3
    transformation of the input.
    """
    # Define input and rotated input
    x = tensor
    Rx = wigner_d_real.transform_tensormap_o3(x)

    # Define the EquivariantLinear module
    f = EquivariantLinear(
        in_keys=x.keys,
        invariant_key_idxs=[i for i, key in enumerate(x.keys) if key["o3_lambda"] == 0],
        in_features=[len(x.block(key).properties) for key in x.keys],
        out_features=[len(x.block(key).properties) - 3 for key in x.keys],
        bias=bias,  # this should only bias the invariant blocks
        dtype=torch.float64,
    )

    # Pass both through the linear layer
    Rfx = wigner_d_real.transform_tensormap_o3(f(x))  # R . f(x)
    fRx = f(Rx)  # f(R . x)

    assert metatensor.allclose(fRx, Rfx, atol=1e-10, rtol=1e-10)
