import argparse
import json
import re
import sys
import time
from collections import namedtuple
from typing import Dict

import numpy as np


BenchmarkResult = namedtuple("BenchmarkResult", ["mean", "std", "min", "max"])


def _do_nothing():
    return []


MIN_ITERS = 20
MAX_ITERS = 1000
MAX_TOTAL_TIME_S = 3


def bench_function(func, setup=None, n_iters=None, n_warmup=None) -> BenchmarkResult:
    """
    Run the given function with the given arguments, and return the time it took to run
    in milliseconds.

    The function will be run ``n_warmup`` times before the actual timing starts, to
    allow for any JIT compilation or other one-time setup to happen before the timing
    starts.

    The function will then be run ``n_iters`` times, and the average time per iteration
    will be returned. If ``n_iters`` is not provided, the function will be run at least
    10 times, and until the total time exceeds 1 second, and the average time per
    iteration will be returned.
    """

    if setup is None:
        setup = _do_nothing

    # warmup
    if n_warmup is not None:
        for _ in range(n_warmup):
            args = setup()
            func(*args)

    # timing
    times = []
    if n_iters is not None:
        for _ in range(n_iters):
            args = setup()
            start_time = time.perf_counter()
            func(*args)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    else:
        total_time = 0
        n_iters = 0
        while n_iters < MIN_ITERS or (
            total_time <= MAX_TOTAL_TIME_S and n_iters <= MAX_ITERS
        ):
            start = time.perf_counter()
            args = setup()

            bench_start = time.perf_counter()
            func(*args)
            bench_end = time.perf_counter()
            times.append(bench_end - bench_start)

            end = time.perf_counter()
            total_time += end - start
            n_iters += 1

    return BenchmarkResult(
        mean=np.mean(times),
        std=np.std(times),
        min=np.min(times),
        max=np.max(times),
    )


def guess_unit(time):
    """Guess the best unit to display the given time in."""
    if time < 1e-6:
        return "ns", 1e9
    elif time < 1e-3:
        return "µs", 1e6
    elif time < 1:
        return "ms", 1e3
    else:
        return "s", 1


def format_benchmark_result(results: Dict[str, BenchmarkResult]) -> str:
    """Format the given benchmark results as a table."""
    header = f"{'Function':<45} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}"
    lines = [header, "-" * len(header)]
    for name, result in results.items():
        time_unit, scale = guess_unit(result.mean)
        lines.append(
            f"{name:<45}"
            f"     {result.mean * scale: >6.2f}{time_unit}"
            f"     {result.std * scale: >6.2f}{time_unit}"
            f"     {result.min * scale: >6.2f}{time_unit}"
            f"     {result.max * scale: >6.2f}{time_unit}"
        )
    return "\n" + "\n".join(lines)


def bench_main(benchmark_name: str, benchmarks: Dict[str, callable]):
    parser = argparse.ArgumentParser(
        description="Run benchmarks and print the results."
    )
    parser.add_argument(
        "benchmarks",
        nargs="*",
        default=[".*"],
        help="set of regex that determine which benchmarks to run",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="output the results as JSON instead of a formatted table",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list all available benchmarks and exit",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="run benchmarks in test mode, with no warmup and only one iteration",
    )
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name in benchmarks.keys():
            print(f" - {name}")
        sys.exit(0)

    if args.test:
        n_warmup = 0
        n_iters = 1
    else:
        n_warmup = 5
        n_iters = None

    results = {}
    interruped = False
    for name, func in benchmarks.items():
        for pattern in args.benchmarks:
            if re.search(pattern, name):
                try:
                    print(f"running {name}...", file=sys.stderr)
                    results[name] = func(n_iters=n_iters, n_warmup=n_warmup)
                    break
                except Exception as e:
                    print(f"error while running {name}: {e}", file=sys.stderr)
                except KeyboardInterrupt:
                    print("benchmark interrupted, stopping...", file=sys.stderr)
                    interruped = True
                    break

        if interruped:
            break

    if args.json:
        print(json.dumps({k: v._asdict() for k, v in results.items()}))
    else:
        print(f"Benchmark results for {benchmark_name}:")
        print(format_benchmark_result(results))
