import sys
import os
import torch

from ._c_lib import _load_library
from . import utils  # noqa


if (sys.version_info.major >= 3) and (sys.version_info.minor >= 8):
    import importlib.metadata

    __version__ = importlib.metadata.version("equistore-torch")

else:
    from pkg_resources import get_distribution

    __version__ = get_distribution("equistore-torch").version

if os.environ.get("EQUISTORE_IMPORT_FOR_SPHINX") is not None:
    from .documentation import Labels, LabelsEntry, TensorBlock, TensorMap
    from .documentation import load, save
else:
    _load_library()
    Labels = torch.classes.equistore.Labels
    LabelsEntry = torch.classes.equistore.LabelsEntry
    TensorBlock = torch.classes.equistore.TensorBlock
    TensorMap = torch.classes.equistore.TensorMap

    load = torch.ops.equistore.load
    save = torch.ops.equistore.save


try:
    import equistore.operations  # noqa

    HAS_EQUISTORE_OPERATIONS = True
except ImportError:
    HAS_EQUISTORE_OPERATIONS = False


if HAS_EQUISTORE_OPERATIONS:
    from . import operations  # noqa
    from .operations import *  # noqa


else:
    # __getattr__ is called when a symbol can not be found, we use it to give
    # the user a better error message if they don't have equistore-operations
    def __getattr__(name):
        raise AttributeError(
            f"equistore.torch.{name} is not defined, are you sure you have the "
            "equistore-operations package installed?"
        )


__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
]
