"""ASV benchmarks for Labels creation and lookup (Python-FFI-Rust stack)."""

import numpy as np

import metatensor


class TimeLabelsCreation:
    """Benchmark Labels construction with uniqueness checking."""

    params = [100, 1_000, 10_000]
    param_names = ["n_entries"]

    def setup(self, n_entries):
        self.names = ["a", "b", "c"]
        self.values = np.column_stack(
            [np.arange(n_entries, dtype=np.int32)] * 3
        )

    def time_create(self, n_entries):
        metatensor.Labels(names=self.names, values=self.values)

    def time_create_assume_unique(self, n_entries):
        metatensor.Labels(
            names=self.names, values=self.values, assume_unique=True
        )


class TimeLabelsLookup:
    """Benchmark Labels.position() lookup performance."""

    params = [1, 100, 1_000]
    param_names = ["n_lookups"]

    def setup(self, n_lookups):
        size = 10_000
        names = ["a", "b", "c"]
        rows = []
        for i in range(size):
            rows.append((i, i, i))
        for i in range(size):
            rows.append((-i, 2 * i, i + 42))
        for i in range(size):
            rows.append((i % 10, i % 49, 3 * i + 9))
        values = np.array(rows, dtype=np.int32)
        # deduplicate to satisfy Labels uniqueness constraint
        values = np.unique(values, axis=0)
        self.labels = metatensor.Labels(names=names, values=values)
        # warm the position hash map
        self.labels.position([0, 0, 0])

    def time_position(self, n_lookups):
        for i in range(n_lookups):
            self.labels.position([i, i % 42, i + 44])
