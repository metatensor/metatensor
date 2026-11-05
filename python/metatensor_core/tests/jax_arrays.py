import io

import numpy as np
import pytest

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def block(values, gradient=True):
    block = TensorBlock(
        values=values,
        samples=Labels.range("s", values.shape[0]),
        components=[Labels.range("c", values.shape[1])],
        properties=Labels.range("p", values.shape[2]),
    )
    if gradient:
        block.add_gradient(
            "g",
            TensorBlock(
                values=2 * values,
                samples=Labels(
                    ["sample", "atom"],
                    np.array([[i, 0] for i in range(values.shape[0])], dtype=np.int32),
                ),
                components=[Labels.range("c", values.shape[1])],
                properties=Labels.range("p", values.shape[2]),
            ),
        )
    return block


def tensor():
    return TensorMap(
        Labels.range("key", 2),
        [
            block(jnp.arange(12, dtype=jnp.float32).reshape(2, 3, 2)),
            block(jnp.arange(12, 24, dtype=jnp.float32).reshape(2, 3, 2)),
        ],
    )


def test_block():
    values = jnp.arange(
        6, dtype=jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    )
    values = values.reshape(1, 3, 2)
    b = block(values)
    assert isinstance(b.values, jax.Array)
    assert b.values.dtype == values.dtype
    assert b.dtype == values.dtype
    assert b.device in values.devices()
    np.testing.assert_equal(np.asarray(b.values), np.asarray(values))
    np.testing.assert_equal(np.asarray(b.gradient("g").values), 2 * np.asarray(values))
    # labels are metadata, they stay NumPy
    assert isinstance(b.samples.values, np.ndarray)


def test_copy():
    b = block(jnp.ones((2, 3, 2)))
    copy = b.copy()
    assert isinstance(copy.values, jax.Array)
    np.testing.assert_equal(np.asarray(copy.values), np.asarray(b.values))


def test_change_backend():
    b = block(jnp.ones((2, 3, 2)))
    as_numpy = b.to(arrays="numpy")
    assert isinstance(as_numpy.values, np.ndarray)
    assert isinstance(as_numpy.gradient("g").values, np.ndarray)

    back = as_numpy.to(arrays="jax")
    assert isinstance(back.values, jax.Array)
    assert isinstance(back.gradient("g").values, jax.Array)
    np.testing.assert_equal(np.asarray(back.values), np.asarray(b.values))


def test_change_device():
    b = block(jnp.ones((2, 3, 2)))
    for device in ("cpu", jax.devices("cpu")[0]):
        for moved in (b.to(device=device), b.to(device)):
            assert isinstance(moved.values, jax.Array)
            assert moved.values.devices() == {jax.devices("cpu")[0]}
            assert isinstance(moved.samples.values, np.ndarray)


def test_change_dtype():
    b = block(jnp.ones((2, 3, 2), dtype=jnp.float32))
    converted = b.to(dtype=jnp.float16)
    assert isinstance(converted.values, jax.Array)
    assert converted.values.dtype == jnp.float16


def test_operations_moving_data():
    # keys_to_samples / keys_to_properties / components_to_properties go through
    # `mts_array_t.create` and `mts_array_t.move_data`, which must handle immutable
    # JAX arrays: compare with the same operations on NumPy arrays
    t = tensor()
    reference = t.to(arrays="numpy")
    for operation, argument in [
        ("keys_to_samples", "key"),
        ("keys_to_properties", "key"),
        ("components_to_properties", "c"),
    ]:
        merged = getattr(t, operation)(argument)
        expected = getattr(reference, operation)(argument)
        for actual, block in zip(merged.blocks(), expected.blocks(), strict=True):
            assert isinstance(actual.values, jax.Array)
            assert actual.samples == block.samples
            assert actual.properties == block.properties
            np.testing.assert_equal(np.asarray(actual.values), block.values)
            np.testing.assert_equal(
                np.asarray(actual.gradient("g").values), block.gradient("g").values
            )


def test_serialization():
    buffer = io.BytesIO()
    metatensor.save(buffer, tensor())
    buffer.seek(0)
    loaded = metatensor.load(buffer)
    for a, b in zip(loaded.blocks(), tensor().blocks(), strict=True):
        np.testing.assert_equal(np.asarray(a.values), np.asarray(b.values))


def assert_same_tensor(actual, expected):
    assert actual.keys == expected.keys
    for a, e in zip(actual.blocks(), expected.blocks(), strict=True):
        assert isinstance(a.values, jax.Array)
        assert a.samples == e.samples
        assert a.components == e.components
        assert a.properties == e.properties
        np.testing.assert_equal(np.asarray(a.values), np.asarray(e.values))
        assert a.gradients_list() == e.gradients_list()
        for parameter in e.gradients_list():
            np.testing.assert_equal(
                np.asarray(a.gradient(parameter).values),
                np.asarray(e.gradient(parameter).values),
            )


def test_pytree():
    t = tensor()
    leaves, treedef = jax.tree.flatten(t)
    # values and gradient of each block
    assert len(leaves) == 4
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert_same_tensor(jax.tree.unflatten(treedef, leaves), t)

    doubled = jax.tree.map(lambda x: 2 * x, t)
    np.testing.assert_equal(
        np.asarray(doubled.block(1).gradient("g").values),
        2 * np.asarray(t.block(1).gradient("g").values),
    )


def test_pytree_info():
    t = tensor()
    t.set_info("unit", "eV")
    assert jax.tree.map(lambda x: 2 * x, t).get_info("unit") == "eV"
    assert jax.jit(lambda t: t)(t).info() == {"unit": "eV"}


def test_pytree_other_leaves():
    t = tensor()
    # JAX can rebuild the pytree with leaves that are not arrays of the right shape,
    # which we can not store in a TensorMap: they are kept in a placeholder instead
    ndims = jax.tree.map(lambda x: x.ndim, t)
    assert jax.tree.leaves(ndims) == [3, 3, 3, 3]
    assert jax.tree.structure(ndims) == jax.tree.structure(t)
    assert repr(ndims) == "TensorMap(<empty>)"

    factors = jax.tree.map(lambda x: 2.0, t)
    doubled = jax.tree.map(lambda x, factor: factor * x, t, factors)
    assert_same_tensor(doubled, jax.tree.map(lambda x: 2 * x, t))

    shapes = jax.eval_shape(lambda t: t.keys_to_properties("key"), t)
    expected = jax.tree.leaves(t.keys_to_properties("key"))
    assert [s.shape for s in jax.tree.leaves(shapes)] == [e.shape for e in expected]


def test_check_grads():
    check_grads = pytest.importorskip("jax.test_util").check_grads

    def loss(t):
        merged = t.keys_to_samples("key").block()
        return (merged.values**2).sum() + merged.gradient("g").values.sum()

    # finite differences need float64
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        check_grads(loss, (tensor().to(dtype=jnp.float64),), order=1, modes=["rev"])
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_jit():
    traces = []

    @jax.jit
    def keys_to_properties(t):
        traces.append(t)
        return t.keys_to_properties("key")

    assert_same_tensor(keys_to_properties(tensor()), tensor().keys_to_properties("key"))

    # metadata is static: equal metadata re-uses the compiled function, and different
    # metadata triggers a new compilation
    keys_to_properties(tensor())
    assert len(traces) == 1
    more_samples = jnp.ones((3, 3, 2))
    keys_to_properties(
        TensorMap(Labels.range("key", 2), [block(more_samples), block(more_samples)])
    )
    assert len(traces) == 2


def test_jit_scattered_moves():
    # both blocks have the same samples, so merging them with keys_to_samples
    # interleaves their rows: the data is moved with index arrays, not slices
    def block_with_samples(values):
        samples = [[system, atom] for system in range(2) for atom in range(3)]
        return TensorBlock(
            values=values,
            samples=Labels(["system", "atom"], np.array(samples, dtype=np.int32)),
            components=[],
            properties=Labels.range("p", values.shape[1]),
        )

    # with x64, JAX keeps the 64-bit indices of the moves as they are, instead of
    # converting (and copying) them to 32-bit
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        t = TensorMap(
            Labels.range("key", 2),
            [
                block_with_samples(jnp.arange(24.0).reshape(6, 4)),
                block_with_samples(jnp.arange(24.0, 48.0).reshape(6, 4)),
            ],
        )
        expected = t.keys_to_samples("key")
        assert_same_tensor(jax.jit(lambda t: t.keys_to_samples("key"))(t), expected)

        expected = t.keys_to_properties("key")
        actual = jax.jit(lambda t: t.keys_to_properties("key"))(t)
        assert_same_tensor(actual, expected)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_grad():
    def loss(values):
        t = TensorMap(Labels.range("key", 2), [block(values), block(2 * values)])
        merged = t.keys_to_samples("key").block()
        return (merged.values**2).sum() + merged.gradient("g").values.sum()

    values = jnp.arange(6, dtype=jnp.float32).reshape(1, 3, 2)
    # d/dv [v^2 + (2v)^2 + 2v + 4v]
    np.testing.assert_allclose(jax.grad(loss)(values), 10 * values + 6)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(values), 10 * values + 6)


def test_vjp():
    values = jnp.arange(6, dtype=jnp.float32).reshape(1, 3, 2)
    output, vjp = jax.vjp(
        lambda v: TensorMap(Labels.range("key", 1), [block(v)]), values
    )
    assert_same_tensor(output, TensorMap(Labels.range("key", 1), [block(values)]))

    cotangent = jax.tree.map(jnp.ones_like, output)
    (gradient,) = vjp(cotangent)
    # values and gradient (2 * values) both contribute
    np.testing.assert_allclose(gradient, 3 * jnp.ones_like(values))


def test_vmap():
    values = jnp.arange(12, dtype=jnp.float32).reshape(2, 1, 3, 2)

    def sum_components(v):
        t = TensorMap(Labels.range("key", 1), [block(v)])
        return t.components_to_properties("c").block().values

    np.testing.assert_equal(
        np.asarray(jax.vmap(sum_components)(values)),
        np.asarray(values).reshape(2, 1, 6),
    )


def test_traced_data():
    message = "can not access the data of a traced JAX array"

    @jax.jit
    def save(t):
        metatensor.save(io.BytesIO(), t)
        return t

    with pytest.raises(ValueError, match=message):
        save(tensor())
