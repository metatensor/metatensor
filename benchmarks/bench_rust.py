"""ASV track_* benchmarks wrapping Criterion (Rust layer).

Runs ``cargo bench --features bench`` once, then extracts
``mean.point_estimate`` (nanoseconds) from Criterion's JSON output
in ``target/criterion/``.
"""

import json
import os
import subprocess


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Map of track method name -> Criterion JSON path (relative to target/criterion/)
_BENCHMARKS = {
    "track_labels_finish_100": "LabelsBuilder/finish/100",
    "track_labels_finish_1000": "LabelsBuilder/finish/1000",
    "track_labels_finish_10000": "LabelsBuilder/finish/10000",
    "track_labels_finish_unique_100": "LabelsBuilder/finish_assume_unique/100",
    "track_labels_finish_unique_1000": "LabelsBuilder/finish_assume_unique/1000",
    "track_labels_finish_unique_10000": "LabelsBuilder/finish_assume_unique/10000",
    "track_labels_lookup_1": "Labels/lookup/1",
    "track_labels_lookup_100": "Labels/lookup/100",
    "track_labels_lookup_1000": "Labels/lookup/1000",
    "track_labels_init_position_3": "Labels/init_position/3",
    "track_labels_init_position_3000": "Labels/init_position/3000",
    "track_labels_init_position_300000": "Labels/init_position/300000",
}


def _read_estimate(criterion_dir):
    """Read mean.point_estimate from a Criterion estimates.json file."""
    path = os.path.join(criterion_dir, "new", "estimates.json")
    if not os.path.exists(path):
        return float("nan")
    with open(path) as f:
        data = json.load(f)
    return data["mean"]["point_estimate"]


class TrackRustBenchmarks:
    """Run Criterion benchmarks and report timings to ASV."""

    timeout = 600

    def setup(self):
        """Run cargo bench once; individual track_* methods read JSON."""
        subprocess.run(
            [
                "cargo", "bench",
                "--features", "bench",
                "-p", "metatensor",
                "--", "--noplot",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        self._criterion_base = os.path.join(REPO_ROOT, "target", "criterion")

    def _get(self, bench_path):
        return _read_estimate(
            os.path.join(self._criterion_base, bench_path)
        )

    def track_labels_finish_100(self):
        return self._get(_BENCHMARKS["track_labels_finish_100"])

    track_labels_finish_100.unit = "ns"

    def track_labels_finish_1000(self):
        return self._get(_BENCHMARKS["track_labels_finish_1000"])

    track_labels_finish_1000.unit = "ns"

    def track_labels_finish_10000(self):
        return self._get(_BENCHMARKS["track_labels_finish_10000"])

    track_labels_finish_10000.unit = "ns"

    def track_labels_finish_unique_100(self):
        return self._get(_BENCHMARKS["track_labels_finish_unique_100"])

    track_labels_finish_unique_100.unit = "ns"

    def track_labels_finish_unique_1000(self):
        return self._get(_BENCHMARKS["track_labels_finish_unique_1000"])

    track_labels_finish_unique_1000.unit = "ns"

    def track_labels_finish_unique_10000(self):
        return self._get(_BENCHMARKS["track_labels_finish_unique_10000"])

    track_labels_finish_unique_10000.unit = "ns"

    def track_labels_lookup_1(self):
        return self._get(_BENCHMARKS["track_labels_lookup_1"])

    track_labels_lookup_1.unit = "ns"

    def track_labels_lookup_100(self):
        return self._get(_BENCHMARKS["track_labels_lookup_100"])

    track_labels_lookup_100.unit = "ns"

    def track_labels_lookup_1000(self):
        return self._get(_BENCHMARKS["track_labels_lookup_1000"])

    track_labels_lookup_1000.unit = "ns"

    def track_labels_init_position_3(self):
        return self._get(_BENCHMARKS["track_labels_init_position_3"])

    track_labels_init_position_3.unit = "ns"

    def track_labels_init_position_3000(self):
        return self._get(_BENCHMARKS["track_labels_init_position_3000"])

    track_labels_init_position_3000.unit = "ns"

    def track_labels_init_position_300000(self):
        return self._get(_BENCHMARKS["track_labels_init_position_300000"])

    track_labels_init_position_300000.unit = "ns"
