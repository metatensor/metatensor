import functools

import torch
from _bench_utils import bench_function, bench_main

from metatensor.torch import Labels, TensorBlock, TensorMap


def bench_labels_small(device, n_iters, n_warmup):
    return bench_function(
        lambda: Labels(
            ["a", "b", "c"],
            torch.tensor(
                [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]], device=device
            ),
        ),
    )


def bench_labels_large(device, n_iters, n_warmup):
    values = torch.tensor(
        [[i, j, k] for i in range(100) for j in range(100) for k in range(100)],
        device=device,
    )
    return bench_function(
        lambda: Labels(["a", "b", "c"], values),
        n_iters=n_iters,
        n_warmup=n_warmup,
    )


def bench_labels_large_assume_unique(device, n_iters, n_warmup):
    values = torch.tensor(
        [[i, j, k] for i in range(100) for j in range(100) for k in range(100)],
        device=device,
    )
    return bench_function(
        lambda: Labels(["a", "b", "c"], values, assume_unique=True),
        n_iters=n_iters,
        n_warmup=n_warmup,
    )


def bench_tensor_block(device, n_iters, n_warmup):
    samples = Labels(
        ["samples"], torch.tensor(range(100), device=device).reshape(-1, 1)
    )
    properties = Labels(
        ["properties"], torch.tensor(range(100), device=device).reshape(-1, 1)
    )
    values = torch.rand(100, 100, device=device)
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


def bench_tensor_block_large(device, n_iters, n_warmup):
    samples = Labels(
        ["samples"], torch.tensor(range(10000), device=device).reshape(-1, 1)
    )
    components = [
        Labels(["component_1"], torch.tensor(range(10), device=device).reshape(-1, 1)),
        Labels(["component_2"], torch.tensor(range(5), device=device).reshape(-1, 1)),
        Labels(["component_3"], torch.tensor(range(10), device=device).reshape(-1, 1)),
    ]
    properties = Labels(
        ["properties"], torch.tensor(range(100), device=device).reshape(-1, 1)
    )
    values = torch.rand(10000, 10, 5, 10, 100, device=device)
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


def bench_tensor_map(device, n_iters, n_warmup):
    n_blocks = 10

    samples = Labels(
        ["samples"], torch.tensor(range(100), device=device).reshape(-1, 1)
    )
    properties = Labels(
        ["properties"], torch.tensor(range(100), device=device).reshape(-1, 1)
    )
    values = torch.rand(100, 100, device=device)

    def _prepare_blocks():
        return (
            [
                TensorBlock(
                    values=values,
                    samples=samples,
                    components=[],
                    properties=properties,
                )
                for _ in range(n_blocks)
            ],
        )

    keys = Labels(["key"], torch.tensor(range(n_blocks), device=device).reshape(-1, 1))

    return bench_function(
        lambda blocks: TensorMap(keys=keys, blocks=blocks),
        setup=_prepare_blocks,
        n_iters=n_iters,
        n_warmup=n_warmup,
    )


def bench_tensor_map_large(device, n_iters, n_warmup):
    n_blocks = 10000

    samples = Labels(
        ["samples"], torch.tensor(range(100), device=device).reshape(-1, 1)
    )
    properties = Labels(
        ["properties"], torch.tensor(range(100), device=device).reshape(-1, 1)
    )
    values = torch.rand(100, 100, device=device)

    def _prepare_blocks():
        return (
            [
                TensorBlock(
                    values=values,
                    samples=samples,
                    components=[],
                    properties=properties,
                )
                for _ in range(n_blocks)
            ],
        )

    keys = Labels(["key"], torch.tensor(range(n_blocks), device=device).reshape(-1, 1))

    return bench_function(
        lambda blocks: TensorMap(keys=keys, blocks=blocks),
        setup=_prepare_blocks,
        n_iters=n_iters,
        n_warmup=n_warmup,
    )


ALL_DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    ALL_DEVICES.append(torch.device("cuda"))
if torch.backends.mps.is_available():
    ALL_DEVICES.append(torch.device("mps"))


if __name__ == "__main__":
    benchmarks = {}
    for device in ALL_DEVICES:
        benchmarks[f"Labels/small ({device.type})"] = functools.partial(
            bench_labels_small, device=device
        )

        benchmarks[f"Labels/large ({device.type})"] = functools.partial(
            bench_labels_large, device=device
        )

        benchmarks[f"Labels/large_assume_unique ({device.type})"] = functools.partial(
            bench_labels_large_assume_unique, device=device
        )

        benchmarks[f"TensorBlock/small ({device.type})"] = functools.partial(
            bench_tensor_block, device=device
        )

        if device.type != "mps":
            # this fails on mps with `'mps.reshape' op the result shape is not
            # compatible with the input shape`
            benchmarks[f"TensorBlock/large ({device.type})"] = functools.partial(
                bench_tensor_block_large, device=device
            )

        benchmarks[f"TensorMap/small ({device.type})"] = functools.partial(
            bench_tensor_map, device=device
        )

        benchmarks[f"TensorMap/large ({device.type})"] = functools.partial(
            bench_tensor_map_large, device=device
        )

    bench_main("Torch Python API", benchmarks)
