# This file defines the default set of classes to use in metatensor-learn
#
# The code from metatensor-learn can be used in two different modes: either pure Python
# mode or TorchScript mode. In the second case, we need to use a different version of
# the classes above; to do so without having to rewrite everything, we re-import
# metatensor-learn in a special context.
#
# See metatensor-torch/metatensor/torch/learn.py for more information.
#
# Any change to this file MUST be also be made to `metatensor/torch/learn.py`.

import re
import warnings

import metatensor


try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


Labels = metatensor.Labels
LabelsEntry = metatensor.LabelsEntry
TensorBlock = metatensor.TensorBlock
TensorMap = metatensor.TensorMap


def torch_jit_is_scripting():
    return False


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
                    "Trying to use code from ``metatensor.learn`` with objects from "
                    "metatensor-torch, you should use the code from "
                    "`metatensor.torch.learn` instead",
                    stacklevel=2,
                )

        return False
