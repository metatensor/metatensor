import argparse
import json
import os
import re
import sys
import time
from collections import namedtuple
from typing import Dict, Optional

import numpy as np


BenchmarkResult = namedtuple(
    "BenchmarkResult", ["samples", "mean", "std", "min", "max"]
)


def _do_nothing():
    return []


MIN_SAMPLES = 20
MAX_TOTAL_TIME_S = 2.0

BENCHMARK_BATCH_THRESHOLD_S = 1e-6
BATCH_SIZE = 100


def _remove_outliers(times):
    """Remove outliers using the IQR method."""
    original_count = len(times)
    if original_count < 4:
        return times
    q1 = np.percentile(times, 25)
    q3 = np.percentile(times, 75)
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    filtered = [t for t in times if lower <= t <= upper]

    if len(filtered) < original_count * 0.5:
        print(
            f"error: more than 50% of samples were outliers "
            f"({original_count - len(filtered)} out of {original_count})",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(filtered) < 1000 and original_count >= 1000:
        print(
            "error: too many outliers, not enough samples left "
            f"({len(filtered)} out of {original_count})",
            file=sys.stderr,
        )
        sys.exit(1)

    return filtered


def bench_function(func, setup=None, n_samples=None, n_warmup=None) -> BenchmarkResult:
    if setup is None:
        setup = _do_nothing

    for _ in range(n_warmup if n_warmup is not None else 5):
        args = setup()
        func(*args)

    args = setup()
    t0 = time.perf_counter()
    func(*args)
    t1 = time.perf_counter()
    use_batching = (t1 - t0) < BENCHMARK_BATCH_THRESHOLD_S

    times = []
    if n_samples is not None:
        for _ in range(n_samples):
            if use_batching:
                all_args = [setup() for _ in range(BATCH_SIZE)]
                t0 = time.perf_counter()
                for args in all_args:
                    func(*args)
                t1 = time.perf_counter()
                times.append((t1 - t0) / BATCH_SIZE)
            else:
                args = setup()
                t0 = time.perf_counter()
                func(*args)
                t1 = time.perf_counter()
                times.append(t1 - t0)
    else:
        total_time = 0
        n = 0
        while n < MIN_SAMPLES or total_time <= MAX_TOTAL_TIME_S:
            if use_batching:
                all_args = [setup() for _ in range(BATCH_SIZE)]
                t0 = time.perf_counter()
                for args in all_args:
                    func(*args)
                t1 = time.perf_counter()
                elapsed = (t1 - t0) / BATCH_SIZE
                times.append(elapsed)
                total_time += t1 - t0
            else:
                t0 = time.perf_counter()
                args = setup()
                bench_start = time.perf_counter()
                func(*args)
                bench_end = time.perf_counter()
                times.append(bench_end - bench_start)
                total_time += time.perf_counter() - t0
            n += 1

    times = _remove_outliers(times)
    return BenchmarkResult(
        samples=len(times),
        mean=np.mean(times),
        std=np.std(times),
        min=np.min(times),
        max=np.max(times),
    )


def _format_samples(samples: int) -> str:
    if samples >= 1_000_000:
        return f"{samples // 1_000_000}M"
    elif samples >= 1_000:
        return f"{samples // 1_000}k"
    else:
        return str(samples)


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


def format_benchmark_result(
    results: Dict[str, BenchmarkResult],
    baseline: Optional[Dict[str, BenchmarkResult]] = None,
) -> str:
    """Format the given benchmark results as a table."""

    header = (
        f"{'Function':<35} {'Samples':>8} {'Mean':>14} {'Std':>14} {'Min':>14} "
        f"{'Max':>14}"
    )
    if baseline is not None:
        header += f" {'vs baseline':>12}"

    lines = [header, "-" * len(header)]
    for name, result in results.items():
        time_unit, scale = guess_unit(result.mean)
        line = (
            f"{name:<35}"
            f"  {_format_samples(result.samples):>7}"
            f"       {result.mean * scale: >6.2f}{time_unit}"
            f"       {result.std * scale: >6.2f}{time_unit}"
            f"       {result.min * scale: >6.2f}{time_unit}"
            f"       {result.max * scale: >6.2f}{time_unit}"
        )
        if baseline is not None and name in baseline:
            ratio = result.mean / baseline[name].mean
            line += f"       {ratio: >6.2f}x"
        elif baseline is not None:
            line += f"       {'N/A':>10}"
        lines.append(line)
    return "\n" + "\n".join(lines)


def _baseline_dir() -> str:
    """Return the directory where baseline files should be stored."""
    return os.environ.get(
        "METATENSOR_BENCHMARK_BASELINE_DIR",
        os.path.dirname(os.path.abspath(sys.argv[0])),
    )


def bench_main(
    benchmark_name: str, benchmarks: Dict[str, callable], baseline_prefix: str = ""
):
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
    parser.add_argument(
        "--save-baseline",
        metavar="NAME",
        help="save benchmark results as a baseline JSON file with the given name",
    )
    parser.add_argument(
        "--baseline",
        metavar="NAME",
        help="load a baseline JSON file and display speedup/slowdown",
    )
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name in benchmarks.keys():
            print(f" - {name}")
        sys.exit(0)

    if args.test:
        n_warmup = 0
        n_samples = 1
    else:
        n_warmup = 5
        n_samples = None

    results = {}
    interruped = False
    for name, func in benchmarks.items():
        for pattern in args.benchmarks:
            if re.search(pattern, name):
                try:
                    print(f"running {name}...", file=sys.stderr)
                    results[name] = func(n_samples=n_samples, n_warmup=n_warmup)
                    break
                except Exception as e:
                    print(f"error while running {name}: {e}", file=sys.stderr)
                except KeyboardInterrupt:
                    print("benchmark interrupted, stopping...", file=sys.stderr)
                    interruped = True
                    break

        if interruped:
            break

    baseline_dir = _baseline_dir()
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir, exist_ok=True)

    if args.save_baseline:
        assert baseline_prefix, "baseline_prefix is required when saving a baseline"
        baseline_name = f"{baseline_prefix}-{args.save_baseline}"
        baseline_path = os.path.join(baseline_dir, baseline_name + ".json")
        with open(baseline_path, "w") as f:
            json.dump({k: v._asdict() for k, v in results.items()}, f, indent=2)

    baseline_data = None
    if args.baseline:
        assert baseline_prefix, "baseline_prefix is required when loading a baseline"
        baseline_name = f"{baseline_prefix}-{args.baseline}"
        baseline_path = os.path.join(baseline_dir, baseline_name + ".json")
        try:
            with open(baseline_path) as f:
                raw = json.load(f)
            baseline_data = {k: BenchmarkResult(**v) for k, v in raw.items()}
        except FileNotFoundError:
            print(
                f"error: could not open baseline file: {baseline_path}", file=sys.stderr
            )

    if args.json:
        output = {k: v._asdict() for k, v in results.items()}
        if baseline_data is not None:
            for name in output:
                if name in baseline_data:
                    output[name]["vs_baseline"] = (
                        results[name].mean / baseline_data[name].mean
                    )
        print(json.dumps(output))
    else:
        print(f"Benchmark results for {benchmark_name}:")
        print(format_benchmark_result(results, baseline_data))
