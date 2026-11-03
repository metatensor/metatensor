import os

import numpy as np
import pytest

import metatensor as mts
from metatensor import Labels, TensorBlock, TensorMap


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
check_grads = pytest.importorskip("jax.test_util").check_grads


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(autouse=True)
def enable_x64():
    # the test data uses float64
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def first(labels, n):
    return Labels(labels.names, labels.values[:n])


def tensor():
    tensor = mts.load(os.path.join(DATA_ROOT, "qm7-power-spectrum.mts"))
    return mts.filter_blocks(tensor, first(tensor.keys, 4))


def square(tensor):
    """Blocks with as many samples as properties, without gradients"""
    blocks = []
    for block in tensor.blocks():
        n = min(block.values.shape)
        values = block.values[:n, :n] + 10 * np.eye(n)
        samples = first(block.samples, n)
        blocks.append(TensorBlock(values, samples, [], first(block.properties, n)))
    return TensorMap(tensor.keys, blocks)


# operations giving a TensorMap (or a list of TensorMap), compared between numpy and
# jax arrays, eagerly and inside `jax.jit`
OPERATIONS = {
    "abs": lambda t: mts.abs(t),
    "add": lambda t: mts.add(t, t),
    "subtract": lambda t: mts.subtract(t, mts.multiply(t, 0.5)),
    "multiply": lambda t: mts.multiply(t, t),
    "divide": lambda t: mts.divide(t, mts.add(mts.abs(t), 1.0)),
    "pow": lambda t: mts.pow(t, 2.0),
    "dot": lambda t: mts.dot(t, mts.remove_gradients(t)),
    "lstsq": lambda t: mts.lstsq(square(t), square(t), rcond=1e-13),
    "solve": lambda t: mts.solve(square(t), square(t)),
    "detach": lambda t: mts.detach(t),
    "remove_gradients": lambda t: mts.remove_gradients(t),
    "requires_grad": lambda t: mts.requires_grad(t, False),
    "zeros_like": lambda t: mts.zeros_like(t),
    "ones_like": lambda t: mts.ones_like(t),
    "make_contiguous": lambda t: mts.make_contiguous(t),
    "drop_blocks": lambda t: mts.drop_blocks(t, first(t.keys, 1)),
    "slice": lambda t: mts.slice(t, "samples", Labels(["atom"], np.array([[0], [2]]))),
    "split": lambda t: mts.split(
        t,
        "properties",
        [Labels(["l"], np.array([[0]])), Labels(["l"], np.array([[1]]))],
    ),
    "sort": lambda t: mts.sort(t, descending=True),
    "join_samples": lambda t: mts.join([t, t], "samples", add_dimension="tensor"),
    "join_properties": lambda t: mts.join([t, t], "properties", add_dimension="tensor"),
    "sum_over_samples": lambda t: mts.sum_over_samples(t, "atom"),
    "mean_over_samples": lambda t: mts.mean_over_samples(t, "atom"),
    "std_over_samples": lambda t: mts.std_over_samples(t, "atom"),
    "var_over_properties": lambda t: mts.var_over_properties(t, "n_1"),
    "std_over_properties": lambda t: mts.std_over_properties(t, "n_1"),
    "insert_dimension": lambda t: mts.insert_dimension(t, "samples", 0, "new", 1),
    "permute_dimensions": lambda t: mts.permute_dimensions(t, "properties", [2, 0, 1]),
}


def assert_same(actual, expected):
    if isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert_same(a, e)
        return

    assert mts.equal_metadata(actual, expected)
    for a, e in zip(actual.blocks(), expected.blocks(), strict=True):
        assert isinstance(a.values, jax.Array)
        np.testing.assert_allclose(np.asarray(a.values), e.values, atol=1e-10)
        for parameter, gradient in e.gradients():
            np.testing.assert_allclose(
                np.asarray(a.gradient(parameter).values), gradient.values, atol=1e-10
            )


@pytest.mark.parametrize("name", OPERATIONS)
def test_operations(name):
    operation = OPERATIONS[name]
    expected = operation(tensor())
    assert_same(operation(tensor().to(arrays="jax")), expected)
    assert_same(jax.jit(operation)(tensor().to(arrays="jax")), expected)


def test_comparisons():
    t = tensor().to(arrays="jax")
    assert mts.equal(t, t)
    assert mts.allclose(t, mts.add(t, 1e-14))
    assert not mts.allclose(t, mts.add(t, 1.0))
    assert mts.equal_metadata(t, t)
    assert mts.is_contiguous(t)
    assert mts.unique_metadata(t, "samples", "system") == mts.unique_metadata(
        tensor(), "samples", "system"
    )


def test_create():
    t = tensor().to(arrays="jax")
    for like in (mts.empty_like(t), mts.random_uniform_like(t)):
        assert mts.equal_metadata(like, t)
        assert all(isinstance(b.values, jax.Array) for b in like.blocks())

    values = jnp.arange(24.0).reshape(2, 3, 4)
    block = mts.block_from_array(values, sample_names=["a", "b"])
    assert isinstance(block.values, jax.Array)
    assert isinstance(block.samples.values, np.ndarray)
    assert block.values.shape == (6, 4)

    jitted = jax.jit(lambda v: mts.block_from_array(v, sample_names=["a", "b"]))
    np.testing.assert_equal(np.asarray(jitted(values).values), np.asarray(block.values))


def random_tensor(seed=0):
    """A tensor with random values and gradients, with more samples per group"""
    rng = np.random.default_rng(seed)
    samples = Labels(
        ["system", "atom"],
        np.array([[s, a] for s in range(3) for a in range(4)], dtype=np.int32),
    )
    properties = Labels(
        ["n", "l"], np.array([[n, m] for n in range(3) for m in range(2)])
    )
    block = TensorBlock(rng.normal(size=(12, 6)), samples, [], properties)
    block.add_gradient(
        "positions",
        TensorBlock(
            rng.normal(size=(24, 3, 6)),
            Labels(
                ["sample", "system", "atom"],
                np.array([[s, s // 4, a] for s in range(12) for a in (0, 1)]),
            ),
            [Labels.range("xyz", 3)],
            properties,
        ),
    )
    return TensorMap(Labels.range("key", 1), [block])


@pytest.mark.parametrize(
    "operation",
    [
        lambda t: mts.multiply(t, t),
        lambda t: mts.dot(t, mts.remove_gradients(t)),
        lambda t: mts.join([t, mts.pow(t, 2.0)], "properties", add_dimension="tensor"),
        lambda t: mts.slice(t, "samples", Labels(["atom"], np.array([[0], [2]]))),
        lambda t: mts.sum_over_samples(t, "atom"),
        lambda t: mts.var_over_samples(t, "atom"),
        lambda t: mts.std_over_samples(t, "atom"),
        lambda t: mts.mean_over_properties(t, "l"),
        lambda t: mts.std_over_properties(t, "l"),
    ],
)
def test_grad(operation):
    reference = random_tensor()

    def loss(arrays):
        values, gradient = arrays
        block = reference.block()
        new_block = TensorBlock(
            jnp.asarray(values), block.samples, block.components, block.properties
        )
        g = block.gradient("positions")
        new_block.add_gradient(
            "positions",
            TensorBlock(jnp.asarray(gradient), g.samples, g.components, g.properties),
        )
        result = operation(TensorMap(reference.keys, [new_block]))
        return sum(
            jnp.sum(b.values**2) + jnp.sum(b.gradient("positions").values ** 2)
            for b in result.blocks()
        )

    block = reference.block()
    arrays = (
        jnp.asarray(block.values),
        jnp.asarray(block.gradient("positions").values),
    )
    check_grads(loss, (arrays,), order=1, modes=["rev"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        jax.jit(jax.grad(loss))(arrays)[0], jax.grad(loss)(arrays)[0]
    )


@pytest.mark.parametrize("operation", ["add", "subtract", "multiply", "divide", "pow"])
def test_scalar_array(operation):
    # a JAX scalar can be a trainable parameter, traced inside jax.jit / jax.grad
    t = mts.add(mts.abs(tensor()), 1.0)
    function = getattr(mts, operation)
    expected = function(t, 1.5)

    t = t.to(arrays="jax")
    assert_same(function(t, jnp.asarray(1.5)), expected)
    assert_same(jax.jit(function)(t, jnp.asarray(1.5)), expected)

    def loss(scalar):
        result = function(t, scalar)
        return sum(
            jnp.sum(b.values) + jnp.sum(b.gradient("positions").values)
            for b in result.blocks()
        )

    check_grads(loss, (jnp.asarray(1.5),), order=2, modes=["fwd", "rev"])
