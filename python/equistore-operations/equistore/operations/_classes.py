# This file defines the default set of classes to use in equistore-operations
#
# The code from equistore-operations can be used in two different modes: either pure
# Python mode or TorchScript mode. In the second case, we need to use a different
# version of the classes above; to do without having to rewrite everything, we re-import
# equistore-operations in a special context.
#
# See equistore-torch/equistore/torch/operations.py for more information.
#
# Any change to this file MUST be also be made to `equistore/torch/operations.py`.

from equistore.core import Labels, TensorBlock, TensorMap


TORCH_SCRIPT_MODE = False
__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
    "TORCH_SCRIPT_MODE",
]
