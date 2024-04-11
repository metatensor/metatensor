import os
import re
from collections import namedtuple

import torch


Version = namedtuple("Version", ["major", "minor", "patch"])


def parse_version(version):
    match = re.match(r"(\d+)\.(\d+)\.(\d+).*", version)
    if match:
        return Version(*map(int, match.groups()))
    else:
        raise ValueError("Invalid version string format")


def version_compatible(actual, required):
    actual = parse_version(actual)
    required = parse_version(required)

    if actual.major != required.major:
        return False
    elif actual.minor != required.minor:
        return False
    else:
        return True


_TORCH_VERSION = parse_version(torch.__version__)

cmake_prefix_path = os.path.join(
    os.path.dirname(__file__),
    f"torch-{_TORCH_VERSION.major}.{_TORCH_VERSION.minor}",
    "lib",
    "cmake",
)
"""
Path containing the CMake configuration files for the underlying C library
"""
