"""ASV benchmarks for TensorMap and Labels serialization."""

import os
import tempfile

import numpy as np

import metatensor
from metatensor.io import load_labels_buffer, save_buffer


def _make_block(n_samples, n_properties, n_components=1):
    """Create a TensorBlock with the given dimensions."""
    values = np.random.default_rng(42).random(
        (n_samples, n_components, n_properties)
    )
    samples = metatensor.Labels(
        names=["sample"],
        values=np.arange(n_samples, dtype=np.int32).reshape(-1, 1),
    )
    components = [
        metatensor.Labels(
            names=["component"],
            values=np.arange(n_components, dtype=np.int32).reshape(-1, 1),
        )
    ]
    properties = metatensor.Labels(
        names=["property"],
        values=np.arange(n_properties, dtype=np.int32).reshape(-1, 1),
    )
    return metatensor.TensorBlock(
        values=values,
        samples=samples,
        components=components,
        properties=properties,
    )


def _make_tensor_map(n_blocks=10, n_samples=1000, n_properties=100):
    """Create a TensorMap with the given number of blocks."""
    keys = metatensor.Labels(
        names=["key"],
        values=np.arange(n_blocks, dtype=np.int32).reshape(-1, 1),
    )
    blocks = [_make_block(n_samples, n_properties) for _ in range(n_blocks)]
    return metatensor.TensorMap(keys=keys, blocks=blocks)


class TimeTensorMapSave:
    """Benchmark TensorMap save to disk."""

    def setup(self):
        self.tensor_map = _make_tensor_map()
        self._tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self._tmpdir, "tensor.mts")

    def teardown(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        os.rmdir(self._tmpdir)

    def time_save(self):
        metatensor.save(self.path, self.tensor_map)


class TimeTensorMapLoad:
    """Benchmark TensorMap load from disk."""

    def setup(self):
        self.tensor_map = _make_tensor_map()
        self._tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self._tmpdir, "tensor.mts")
        metatensor.save(self.path, self.tensor_map)

    def teardown(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        os.rmdir(self._tmpdir)

    def time_load(self):
        metatensor.load(self.path)


class TimeLabelsSerialize:
    """Benchmark Labels in-memory serialization round-trip."""

    def setup(self):
        n_entries = 10_000
        names = ["a", "b", "c"]
        values = np.column_stack(
            [np.arange(n_entries, dtype=np.int32)] * 3
        )
        self.labels = metatensor.Labels(names=names, values=values)
        self.buffer = save_buffer(self.labels)

    def time_save_buffer(self):
        save_buffer(self.labels)

    def time_load_buffer(self):
        load_labels_buffer(self.buffer)
