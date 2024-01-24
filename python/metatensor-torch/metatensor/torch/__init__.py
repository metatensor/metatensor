import os
import torch

from .version import __version__  # noqa
from ._c_lib import _load_library
from . import utils  # noqa


if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX") is not None:
    from .documentation import Labels, LabelsEntry, TensorBlock, TensorMap
    from .documentation import load, load_labels, load_labels_buffer, load_buffer
    from .documentation import save, save_buffer
else:
    _load_library()
    Labels = torch.classes.metatensor.Labels
    LabelsEntry = torch.classes.metatensor.LabelsEntry
    TensorBlock = torch.classes.metatensor.TensorBlock
    TensorMap = torch.classes.metatensor.TensorMap

    load = torch.ops.metatensor.load
    load_buffer = torch.ops.metatensor.load_buffer
    load_labels = torch.ops.metatensor.load_labels
    load_labels_buffer = torch.ops.metatensor.load_labels_buffer
    save = torch.ops.metatensor.save
    save_buffer = torch.ops.metatensor.save_buffer

from . import atomistic  # noqa

MISSING_SUBPACKAGES = []

try:
    import metatensor.operations  # noqa

    HAS_METATENSOR_OPERATIONS = True
except ImportError:
    HAS_METATENSOR_OPERATIONS = False


if HAS_METATENSOR_OPERATIONS:
    from . import operations  # noqa
    from .operations import *  # noqa
else:
    MISSING_SUBPACKAGES.append("metatensor-operations")


try:
    import metatensor.learn  # noqa

    HAS_METATENSOR_LEARN = True
except ImportError:
    HAS_METATENSOR_LEARN = False


if HAS_METATENSOR_LEARN:
    from . import learn  # noqa
    from .learn import *  # noqa
else:
    MISSING_SUBPACKAGES.append("metatensor-learn")


if not (HAS_METATENSOR_LEARN) or not (HAS_METATENSOR_OPERATIONS):
    # __getattr__ is called when a symbol can not be found, we use it to give
    # the user a better error message if they don't have all metatensor subpackages
    def __getattr__(name):
        raise AttributeError(
            f"metatensor.torch.{name} is not defined, you might have not "
            "installed the corresponding metatensor subpackage "
            f"({' or '.join(MISSING_SUBPACKAGES)})."
        )


__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
]
