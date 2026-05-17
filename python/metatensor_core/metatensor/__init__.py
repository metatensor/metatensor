# The metatensor python package is distributed in multiple pieces, each with its own
# version number and dependency management:
#
# - the `metatensor-core` distribution contains the Python bindings to the
#   metatensor-core C API in the `metatensor` python package. External package
#   integrating with metatensor such as featomic only depend on metatensor-core;
# - the `metatensor-operations` distribution contains the operations to manipulate
#   metatensor data;
# - the `metatensor-learn` distribution contains code to define custom machine learning
#   models and train them;
# - the `metatensor-torch` distribution contains the TorchScript bindings to the C API.
#   it installs itself in `metatensor/torch`.
#
# There is also a `metatensor` distribution which does not contain any package, only
# declaring dependencies on `metatensor-core`, `metatensor-operation`, and
# `metatensor-learn`; and an optional dependency on `metatensor-torch`.

from . import utils  # noqa: F401
from ._block import TensorBlock  # noqa: F401
from ._data import (  # noqa: F401
    Array,
    DeviceWarning,
    ExternalCpuArray,
    ExternalCudaArray,
    register_external_data_wrapper,
)
from ._labels import Labels, LabelsEntry
from ._status import MetatensorError
from ._tensor import TensorMap
from ._version import __version__  # noqa: F401
from .io import (  # noqa: F401
    load,
    load_block,
    load_block_mmap,
    load_labels,
    load_mmap,
    save,
)


# pretend the classes are defined in the top-level module for better error messages
Labels.__module__ = __name__
LabelsEntry.__module__ = __name__
TensorBlock.__module__ = __name__
TensorMap.__module__ = __name__
MetatensorError.__module__ = __name__
DeviceWarning.__module__ = __name__
ExternalCpuArray.__module__ = __name__
ExternalCudaArray.__module__ = __name__

try:
    import metatensor_operations  # noqa: F401

    HAS_METATENSOR_OPERATIONS = True
except ImportError:
    HAS_METATENSOR_OPERATIONS = False


if HAS_METATENSOR_OPERATIONS:
    from . import operations  # noqa: F401
    from .operations import *  # noqa: F401, F403


# __getattr__ is called when a module attribute can not be found. We use it to provide
# access to symbols from metatensor-operations even if this module was imported while
# metatensor-operations was still initializing, and to provide a clearer error message
# when metatensor-operations is not installed.
def __getattr__(name):
    if HAS_METATENSOR_OPERATIONS:
        from . import operations

        try:
            # Mimic the behavior of "from .operations import *"
            value = getattr(operations, name)
            globals()[name] = value
            return value
        except AttributeError:
            pass

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
