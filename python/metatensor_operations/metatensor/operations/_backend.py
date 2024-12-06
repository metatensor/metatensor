# This file defines the default set of classes to use in metatensor-operations
#
# The code from metatensor-operations can be used in two different modes: either pure
# Python mode or TorchScript mode. In the second case, we need to use a different
# version of the classes above; to do without having to rewrite everything, we re-import
# metatensor-operations in a special context.
#
# See metatensor-torch/metatensor/torch/operations.py for more information.
#
# Any change to this file MUST be also be made to `metatensor/torch/operations.py`.
import re
import warnings
from typing import Union

import numpy as np

import metatensor


try:
    import torch

    Array = Union[np.ndarray, torch.Tensor]
    _HAS_TORCH = True
except ImportError:
    Array = np.ndarray
    _HAS_TORCH = False


Labels = metatensor.Labels
TensorBlock = metatensor.TensorBlock
TensorMap = metatensor.TensorMap

# type used by the values in Labels
LabelsValues = np.ndarray


def torch_jit_is_scripting():
    return False


def torch_jit_script(function):
    """
    This function takes the place of ``@torch.jit.script`` when running in pure Python
    mode (i.e. when using classes from metatensor-core).

    When used as a decorator (``@torch_jit_script def foo()``), it does nothing to the
    function.
    """
    return function


def torch_jit_annotate(type, value):
    """
    This function takes the place of ``torch.jit.annotate`` when running in pure Python
    mode (i.e. when using classes from metatensor-core).
    """
    return value


_VERSION_REGEX = re.compile(r"(\d+)\.(\d+)\.*.")


def _version_at_least(version, expected):
    version = tuple(map(int, _VERSION_REGEX.match(version).groups()))
    expected = tuple(map(int, _VERSION_REGEX.match(expected).groups()))

    return version >= expected


def is_metatensor_class(value, typ):
    assert typ in (Labels, TensorBlock, TensorMap)

    if isinstance(value, typ):
        return True
    else:
        if _HAS_TORCH and isinstance(value, torch.ScriptObject):
            if _version_at_least(torch.__version__, "2.1.0"):
                # _type() is only working for torch >= 2.1
                is_metatensor_torch_class = "metatensor" in str(value._type())
            else:
                # we don't know, it's fine
                is_metatensor_torch_class = False

            if is_metatensor_torch_class:
                warnings.warn(
                    "Trying to use operations from metatensor with objects from "
                    "metatensor-torch, you should use the operation from "
                    "`metatensor.torch` as well, e.g. `metatensor.torch.add(...)` "
                    "instead of `metatensor.add(...)`",
                    stacklevel=2,
                )

        return False
