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


from .version import __version__  # noqa
from . import utils  # noqa
from .block import TensorBlock  # noqa
from .labels import Labels, LabelsEntry  # noqa
from .status import MetatensorError  # noqa
from .tensor import TensorMap  # noqa

from .io import load, save  # noqa

try:
    from . import operations  # noqa

    HAS_METATENSOR_OPERATIONS = True
except ImportError:
    HAS_METATENSOR_OPERATIONS = False


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


__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
]
