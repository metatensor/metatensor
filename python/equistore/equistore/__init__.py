# The equistore python package is distributed in multiple pieces, each with its
# own version number and dependency management:
#
# - the `equistore` package contains only this file, and is the default
#   user-facing package. It depends on the other packages, and re-export all
#   classes/functions from them
# - the `equistore-core` package contains the Python bindings to the
#   equistore-core C API. Is is using Python's namespace packages
#   (https://peps.python.org/pep-0420/) to install itself in `equistore/core`.
#   The version of this package is the same as the version of the equistore-core
#   rust crate. External package integrating with equistore such as rascaline
#   only depend on equistore-core.
# - the `equistore-operations` package contains the operations to manipulate
#   equistore data.

import sys

if (sys.version_info.major >= 3) and (sys.version_info.minor >= 8):
    import importlib.metadata

    __version__ = importlib.metadata.version("equistore")

else:
    from pkg_resources import get_distribution

    __version__ = get_distribution("equistore").version

from . import core  # noqa

try:
    from . import operations  # noqa

    HAS_EQUISTORE_OPERATIONS = True
except ImportError:
    HAS_EQUISTORE_OPERATIONS = False


from .core import *  # noqa

if HAS_EQUISTORE_OPERATIONS:
    from .operations import *  # noqa

else:
    # __getattr__ is called when a symbol can not be found, we use it to give
    # the user a better error message if they don't have equistore-operations
    def __getattr__(name):
        raise AttributeError(
            f"equistore.{name} is not defined, are you sure you have the "
            "equistore-operations package installed?"
        )


def print_versions():
    print(f"equistore v{__version__} using:")
    print(f"  - equistore-core v{core.__version__}")
    if HAS_EQUISTORE_OPERATIONS:
        print(f"  - equistore-operations v{operations.__version__}")
    else:
        print("  - equistore-operations is not installed")
