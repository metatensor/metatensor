# The metatensor python package is distributed in multiple pieces, each with its own
# version number and dependency management:
#
# - the `metatensor-core` distribution contains the Python bindings to the
#   metatensor-core C API in the `metatensor` python package. External package
#   integrating with metatensor such as featomic only depend on metatensor-core.
# - the `metatensor-operations` distribution contains the operations to manipulate
#   metatensor data. Is is using Python's namespace packages
#   (https://peps.python.org/pep-0420/) to install itself in `metatensor/operations`.
# - the `metatensor-torch` distribution contains the TorchScript bindings to the C API.
#   it installs itself in `metatensor/torch`.
#
# There is also a `metatensor` distribution which does not contain any package, and
# only declares dependencies on `metatensor-core` and `metatensor-operation`; as well
# an an optional dependency on `metatensor-torch`.

import importlib
import sys
import warnings

from . import utils  # noqa: F401
from .block import TensorBlock  # noqa: F401
from .data import DeviceWarning  # noqa: F401
from .io import load, load_block, load_labels, save  # noqa: F401
from .labels import Labels, LabelsEntry  # noqa: F401
from .status import MetatensorError  # noqa: F401
from .tensor import TensorMap  # noqa: F401
from .version import __version__  # noqa: F401


# Define the map between old namespace paths and new flat packages
_PACKAGE_MAP = {
    "metatensor": "metatensor_core",
    "metatensor.operations": "metatensor_operations",
    "metatensor.learn": "metatensor_learn",
    "metatensor.torch": "metatensor_torch",
}

# Register aliases in sys.modules to allow `import metatensor.torch` syntax
for namespace_path, flat_name in _PACKAGE_MAP.items():
    try:
        # Attempt to import the actual flat package
        pkg = importlib.import_module(flat_name)
        # Alias it in sys.modules so python thinks metatensor.x is loaded
        sys.modules[namespace_path] = pkg
    except ImportError as e:
        # Don't fail hard if optional packages are missing
        if "No module named" not in str(e):
            warnings.warn(f"Failed to import {flat_name}: {e}", stacklevel=2)

# Re-export operations symbols into the top-level namespace
if "metatensor.operations" in sys.modules:
    from metatensor_operations import *  # noqa: F401, F403


def __getattr__(name):
    # Handle attribute access (e.g. metatensor.torch) via the map
    # This catches cases where the import wasn't explicit
    full_name = f"metatensor.{name}"
    if full_name in _PACKAGE_MAP:
        flat_name = _PACKAGE_MAP[full_name]
        try:
            return importlib.import_module(flat_name)
        except ImportError as e:
            raise AttributeError(
                f"metatensor.{name} is not defined, are you sure you have the "
                f"{flat_name.replace('_', '-')} package installed?"
            ) from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
]
