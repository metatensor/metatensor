import sys
import os
import torch

from ._c_lib import _load_library
from . import utils  # noqa


if (sys.version_info.major >= 3) and (sys.version_info.minor >= 8):
    import importlib.metadata

    __version__ = importlib.metadata.version("metatensor-torch")

else:
    from pkg_resources import get_distribution

    __version__ = get_distribution("metatensor-torch").version

if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX") is not None:
    from .documentation import Labels, LabelsEntry, TensorBlock, TensorMap
    from .documentation import load, save
else:
    _load_library()
    Labels = torch.classes.metatensor.Labels
    LabelsEntry = torch.classes.metatensor.LabelsEntry
    TensorBlock = torch.classes.metatensor.TensorBlock
    TensorMap = torch.classes.metatensor.TensorMap

    load = torch.ops.metatensor.load
    save = torch.ops.metatensor.save


try:
    import metatensor.operations  # noqa

    HAS_METATENSOR_OPERATIONS = True
except ImportError:
    HAS_METATENSOR_OPERATIONS = False


if HAS_METATENSOR_OPERATIONS:
    from . import operations  # noqa
    from .operations import *  # noqa


else:
    # __getattr__ is called when a symbol can not be found, we use it to give
    # the user a better error message if they don't have metatensor-operations
    def __getattr__(name):
        raise AttributeError(
            f"metatensor.torch.{name} is not defined, are you sure you have the "
            "metatensor-operations package installed?"
        )


__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
]
