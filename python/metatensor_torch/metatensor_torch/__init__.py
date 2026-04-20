import os
import sys
from typing import TYPE_CHECKING

import torch

import metatensor


sys.modules["metatensor.torch"] = sys.modules[__name__]
if not hasattr(metatensor, "torch"):
    metatensor.torch = sys.modules[__name__]


from . import utils  # noqa: E402, F401
from ._c_lib import _load_library  # noqa: E402
from ._version import __version__  # noqa: E402, F401


if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX", "0") != "0" or TYPE_CHECKING:
    from ._documentation import (
        Labels,
        LabelsEntry,
        TensorBlock,
        TensorMap,
        dtype_name,
        load_block_buffer,
        load_buffer,
        load_labels_buffer,
        save_buffer,
        version,
    )
else:
    _load_library()
    Labels = torch.classes.metatensor.Labels
    LabelsEntry = torch.classes.metatensor.LabelsEntry
    TensorBlock = torch.classes.metatensor.TensorBlock
    TensorMap = torch.classes.metatensor.TensorMap

    version = torch.ops.metatensor.version
    dtype_name = torch.ops.metatensor.dtype_name
    load_buffer = torch.ops.metatensor.load_buffer
    load_block_buffer = torch.ops.metatensor.load_block_buffer
    load_labels_buffer = torch.ops.metatensor.load_labels_buffer
    save_buffer = torch.ops.metatensor.save_buffer

from .io import (  # noqa: F401, E402
    load,
    load_block,
    load_labels,
    save,
)


try:
    import metatensor_operations  # noqa: F401, E402

    HAS_METATENSOR_OPERATIONS = True
except ImportError:
    HAS_METATENSOR_OPERATIONS = False

if HAS_METATENSOR_OPERATIONS:
    from . import operations  # noqa: F401
    from .operations import *  # noqa: F401, F403

    sys.modules["metatensor.torch.operations"] = operations
    metatensor.torch.operations = operations

    sys.modules["metatensor.torch.operations._backend"] = operations._backend
    metatensor.torch.operations._backend = operations._backend
else:
    # __getattr__ is called when a module attribute can not be found, we use it to
    # give the user a better error message if they don't have metatensor-operations
    def __getattr__(name):
        raise AttributeError(
            f"metatensor.torch.{name} is not defined, are you sure you have the "
            "metatensor-operations package installed?"
        )


try:
    import metatensor_learn  # noqa: F401

    from . import learn  # noqa: F401

    sys.modules["metatensor.torch.learn"] = learn
    metatensor.torch.learn = learn

    sys.modules["metatensor.torch.learn._backend"] = learn._backend
    metatensor.torch.learn._backend = learn._backend
except ImportError:
    pass


from . import _module  # noqa: E402


_module.patch_torch_jit_module()

__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
]
