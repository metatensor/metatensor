# The metatensor python package is distributed in multiple pieces, each with its
# own version number and dependency management:
#
# - the `metatensor` package contains only this file, and is the default
#   user-facing package. It depends on the other packages, and re-export all
#   classes/functions from them
# - the `metatensor-core` package contains the Python bindings to the
#   metatensor-core C API. Is is using Python's namespace packages
#   (https://peps.python.org/pep-0420/) to install itself in `metatensor/core`.
#   The version of this package is the same as the version of the metatensor-core
#   rust crate. External package integrating with metatensor such as rascaline
#   only depend on metatensor-core.
# - the `metatensor-operations` package contains the operations to manipulate
#   metatensor data.

import sys

if (sys.version_info.major >= 3) and (sys.version_info.minor >= 8):
    import importlib.metadata

    __version__ = importlib.metadata.version("metatensor")

else:
    from pkg_resources import get_distribution

    __version__ = get_distribution("metatensor").version

from . import core  # noqa

try:
    from . import operations  # noqa

    HAS_METATENSOR_OPERATIONS = True
except ImportError:
    HAS_METATENSOR_OPERATIONS = False


from .core import *  # noqa

if HAS_METATENSOR_OPERATIONS:
    from .operations import *  # noqa

else:
    # __getattr__ is called when a symbol can not be found, we use it to give
    # the user a better error message if they don't have metatensor-operations
    def __getattr__(name):
        raise AttributeError(
            f"metatensor.{name} is not defined, are you sure you have the "
            "metatensor-operations package installed?"
        )


def print_versions():
    print(f"metatensor v{__version__} using:")
    print(f"  - metatensor-core v{core.__version__}")
    if HAS_METATENSOR_OPERATIONS:
        print(f"  - metatensor-operations v{operations.__version__}")
    else:
        print("  - metatensor-operations is not installed")
