# The metatensor python package is distributed in multiple pieces, each with its own
# version number and dependency management:
#
# - the `metatensor-core` distribution contains the Python bindings to the
#   metatensor-core C API in the `metatensor` python package. External package
#   integrating with metatensor such as rascaline only depend on metatensor-core.
# - the `metatensor-operations` distribution contains the operations to manipulate
#   metatensor data. Is is using Python's namespace packages
#   (https://peps.python.org/pep-0420/) to install itself in `metatensor/operations`.
# - the `metatensor-torch` distribution contains the TorchScript bindings to the C API.
#   it installs itself in `metatensor/torch`.
#
# There is also a `metatensor` distribution which does not contain any package, and
# only declares dependencies on `metatensor-core` and `metatensor-operation`; as well
# an an optional dependency on `metatensor-torch`.

from .version import __version__  # noqa: F401
from . import utils  # noqa: F401
from .block import TensorBlock  # noqa: F401
from .labels import Labels, LabelsEntry  # noqa: F401
from .status import MetatensorError  # noqa: F401
from .tensor import TensorMap  # noqa: F401
from .data import DeviceWarning  # noqa: F401

from .io import load, load_labels, save  # noqa: F401

try:
    from . import operations  # noqa: F401

    HAS_METATENSOR_OPERATIONS = True
except ImportError:
    HAS_METATENSOR_OPERATIONS = False


if HAS_METATENSOR_OPERATIONS:
    from .operations import *  # noqa: F401, F403

else:
    # __getattr__ is called when a module attribute can not be found, we use it to give
    # the user a better error message if they don't have metatensor-operations
    def __getattr__(name):
        raise AttributeError(
            f"metatensor.{name} is not defined, are you sure you have the "
            "metatensor-operations package installed?"
        )


try:
    from . import learn  # noqa: F401
except ImportError:
    pass


__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
]
