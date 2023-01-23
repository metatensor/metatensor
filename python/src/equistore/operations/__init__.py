"""
The Python API for equistore also provides functions which operate on
:py:class:`equistore.TensorMap`, and can be used to build Machine Learning
models.

These functions can handle data stored either in numpy arrays or Torch tensor,
and automatically dispatch to the right function for a given TensorMap.
"""

from .allclose import (  # noqa
    allclose,
    allclose_block,
    allclose_block_raise,
    allclose_raise,
)
from .dot import dot  # noqa
from .empty_like import empty_like, empty_like_block  # noqa
from .join import join  # noqa
from .lstsq import lstsq  # noqa
from .ones_like import ones_like, ones_like_block  # noqa
from .reduce_over_samples import mean_over_samples, sum_over_samples  # noqa
from .remove_gradients import remove_gradients  # noqa
from .slice import slice, slice_block  # noqa
from .solve import solve  # noqa
from .zeros_like import zeros_like, zeros_like_block  # noqa


__all__ = [
    "allclose",
    "allclose_raise",
    "allclose_block",
    "allclose_block_raise",
    "empty_like",
    "dot",
    "join",
    "lstsq",
    "mean_over_samples",
    "ones_like",
    "remove_gradients",
    "slice",
    "slice_block",
    "solve",
    "sum_over_samples",
    "zeros_like",
]
