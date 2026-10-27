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


def test_traced_values():
    message = "can not store a traced JAX value in metatensor"

    @jax.jit
    def create(values):
        return block(values, gradient=False).values

    with pytest.raises(TypeError, match=message):
        create(jnp.ones((1, 1, 1)))


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
        moved = b.to(device=device)
        assert isinstance(moved.values, jax.Array)
        assert moved.values.devices() == {jax.devices("cpu")[0]}


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
