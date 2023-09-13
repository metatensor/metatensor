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

from metatensor import Labels, TensorBlock, TensorMap


def torch_jit_is_scripting():
    return False


check_isinstance = isinstance

__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
    "torch_jit_is_scripting",
    "check_isinstance",
]
