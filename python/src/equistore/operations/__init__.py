"""
The Python API for equistore also provides functions which operate on
:py:class:`equistore.TensorMap`, :py:class:`equistore.TensorBlocks` as well as
:py:class:`equistore.Labels` and can be used to build Machine Learning
models.

These functions can handle data stored either in numpy arrays or Torch tensor,
and automatically dispatch to the right function for a given TensorMap.
"""
from .add import add  # noqa
from .allclose import (  # noqa
    allclose,
    allclose_block,
    allclose_block_raise,
    allclose_raise,
)
from .divide import divide  # noqa
from .dot import dot  # noqa
from .drop_blocks import drop_blocks
from .empty_like import empty_like, empty_like_block  # noqa
from .equal import equal, equal_block, equal_block_raise, equal_raise  # noqa
from .equal_metadata import equal_metadata  # noqa
from .join import join  # noqa
from .lstsq import lstsq  # noqa
from .multiply import multiply  # noqa
from .ones_like import ones_like, ones_like_block  # noqa
from .pow import pow  # noqa
from .reduce_over_samples import (  # noqa
    mean_over_samples,
    std_over_samples,
    sum_over_samples,
    variance_over_samples,
)
from .remove_gradients import remove_gradients  # noqa
from .slice import slice, slice_block  # noqa
from .solve import solve  # noqa
from .split import split, split_block  # nopa
from .subtract import subtract  # noqa
from .unique_metadata import unique_metadata, unique_metadata_block  # noqa
from .zeros_like import zeros_like, zeros_like_block  # noqa


__all__ = [
    "add",
    "allclose",
    "allclose_raise",
    "allclose_block",
    "allclose_block_raise",
    "divide",
    "dot",
    "drop_blocks",
    "empty_like",
    "empty_like_block",
    "equal",
    "equal_raise",
    "equal_block",
    "equal_block_raise",
    "equal_metadata",
    "join",
    "lstsq",
    "mean_over_samples",
    "ones_like",
    "ones_like_block",
    "multiply",
    "pow",
    "remove_gradients",
    "slice",
    "slice_block",
    "std_over_samples",
    "solve",
    "split",
    "split_block",
    "subtract",
    "sum_over_samples",
    "unique_metadata",
    "unique_metadata_block",
    "variance_over_samples",
    "zeros_like",
    "zeros_like_block",
]
