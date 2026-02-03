import importlib
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


_HERE = os.path.dirname(__file__)
_TORCH_VERSION = parse_version(torch.__version__)
install_prefix = os.path.join(
    _HERE, f"torch-{_TORCH_VERSION.major}.{_TORCH_VERSION.minor}"
)


external_path = os.path.join(install_prefix, "_external.py")
if os.path.exists(external_path):
    spec = importlib.util.spec_from_file_location("_external", external_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cmake_prefix_path = module.EXTERNAL_METATENSOR_TORCH_PREFIX
    """
    Path containing the CMake configuration files for the underlying C++ library
    """
else:
    cmake_prefix_path = os.path.join(install_prefix, "lib", "cmake")
    """
    Path containing the CMake configuration files for the underlying C++ library
    """
