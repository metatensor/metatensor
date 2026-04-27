import numpy as np
from _bench_utils import bench_function, bench_main

from metatensor import Labels, TensorBlock, TensorMap


def bench_labels_small(n_iters, n_warmup):
    return bench_function(
        lambda: Labels(
            ["a", "b", "c"],
            np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]]),
        ),
        n_iters=n_iters,
        n_warmup=n_warmup,
    )


def bench_labels_large(n_iters, n_warmup):
    values = np.array(
        [[i, j, k] for i in range(100) for j in range(100) for k in range(100)]
    )
    return bench_function(
        lambda: Labels(["a", "b", "c"], values),
        n_iters=n_iters,
        n_warmup=n_warmup,
    )


def bench_labels_large_assume_unique(n_iters, n_warmup):
    values = np.array(
        [[i, j, k] for i in range(100) for j in range(100) for k in range(100)]
    )
    return bench_function(
        lambda: Labels(["a", "b", "c"], values, assume_unique=True),
        n_iters=n_iters,
        n_warmup=n_warmup,
    )


def bench_tensor_block(n_iters, n_warmup):
    samples = Labels(["samples"], np.array(range(100)).reshape(-1, 1))
    properties = Labels(["properties"], np.array(range(100)).reshape(-1, 1))
    values = np.random.rand(100, 100)
    return bench_function(
        lambda: TensorBlock(
            values=values,
            samples=samples,
            components=[],
            properties=properties,
        ),
        n_iters=n_iters,
        n_warmup=n_warmup,
    )


def bench_tensor_block_large(n_iters, n_warmup):
    samples = Labels(["samples"], np.array(range(10000)).reshape(-1, 1))
    components = [
        Labels(["component_1"], np.array(range(10)).reshape(-1, 1)),
        Labels(["component_2"], np.array(range(5)).reshape(-1, 1)),
        Labels(["component_3"], np.array(range(10)).reshape(-1, 1)),
    ]
    properties = Labels(["properties"], np.array(range(100)).reshape(-1, 1))
    values = np.random.rand(10000, 10, 5, 10, 100)
    return bench_function(
        lambda: TensorBlock(
            values=values,
            samples=samples,
            components=components,
            properties=properties,
        ),
        n_iters=n_iters,
        n_warmup=n_warmup,
    )


def bench_tensor_map(n_iters, n_warmup):
    n_blocks = 10

    def _prepare_blocks():
        blocks = []
        for _ in range(n_blocks):
            samples = Labels(["samples"], np.array(range(100)).reshape(-1, 1))
            properties = Labels(["properties"], np.array(range(100)).reshape(-1, 1))
            values = np.random.rand(100, 100)
            block = TensorBlock(
                values=values,
                samples=samples,
                components=[],
                properties=properties,
            )
            blocks.append(block)
        return (blocks,)

    keys = Labels(["key"], np.array(range(n_blocks)).reshape(-1, 1))

    return bench_function(
        lambda blocks: TensorMap(keys=keys, blocks=blocks),
        setup=_prepare_blocks,
        n_iters=n_iters,
        n_warmup=n_warmup,
    )


def bench_tensor_map_large(n_iters, n_warmup):
    n_blocks = 10000

    def _prepare_blocks():
        blocks = []
        for _ in range(n_blocks):
            samples = Labels(["samples"], np.array(range(100)).reshape(-1, 1))
            properties = Labels(["properties"], np.array(range(100)).reshape(-1, 1))
            values = np.random.rand(100, 100)
            block = TensorBlock(
                values=values,
                samples=samples,
                components=[],
                properties=properties,
            )
            blocks.append(block)
        return (blocks,)

    keys = Labels(["key"], np.array(range(n_blocks)).reshape(-1, 1))

    return bench_function(
        lambda blocks: TensorMap(keys=keys, blocks=blocks),
        setup=_prepare_blocks,
        n_iters=n_iters,
        n_warmup=n_warmup,
    )


if __name__ == "__main__":
    benchmarks = {
        "Labels/small": bench_labels_small,
        "Labels/large": bench_labels_large,
        "Labels/large_assume_unique": bench_labels_large_assume_unique,
        "TensorBlock/small": bench_tensor_block,
        "TensorBlock/large": bench_tensor_block_large,
        "TensorMap/small": bench_tensor_map,
        "TensorMap/large": bench_tensor_map_large,
    }

    bench_main("Python API", benchmarks)
