import os
from typing import TYPE_CHECKING

import torch

from . import utils  # noqa: F401
from ._c_lib import _load_library
from .version import __version__  # noqa: F401


if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX", "0") != "0" or TYPE_CHECKING:
    from .documentation import (
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

from .serialization import (  # noqa: F401
    load,
    load_block,
    load_labels,
    save,
)


try:
    import metatensor.operations  # noqa: F401

    HAS_METATENSOR_OPERATIONS = True
except ImportError:
    HAS_METATENSOR_OPERATIONS = False

if HAS_METATENSOR_OPERATIONS:
    from . import operations  # noqa: F401
    from .operations import *  # noqa: F401, F403
else:
    # __getattr__ is called when a module attribute can not be found, we use it to
    # give the user a better error message if they don't have metatensor-operations
    def __getattr__(name):
        raise AttributeError(
            f"metatensor.torch.{name} is not defined, are you sure you have the "
            "metatensor-operations package installed?"
        )


try:
    import metatensor.learn  # noqa: F401

    from . import learn  # noqa: F401

except ImportError:
    pass


from . import atomistic  # noqa: F401
from ._module import _patch_torch_jit_module


_patch_torch_jit_module()

__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
]
