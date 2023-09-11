import sys

if (sys.version_info.major >= 3) and (sys.version_info.minor >= 8):
    import importlib.metadata

    __version__ = importlib.metadata.version("metatensor-operations")

else:
    from pkg_resources import get_distribution

    __version__ = get_distribution("metatensor-operations").version


from ._utils import NotEqualError
from .abs import abs
from .add import add
from .allclose import (
    allclose,
    allclose_block,
    allclose_block_raise,
    allclose_raise,
)
from .block_from_array import block_from_array
from .divide import divide
from .dot import dot
from .drop_blocks import drop_blocks
from .empty_like import empty_like, empty_like_block
from .equal import equal, equal_block, equal_block_raise, equal_raise
from .equal_metadata import (
    equal_metadata,
    equal_metadata_block,
    equal_metadata_raise,
    equal_metadata_block_raise,
)
from .join import join
from .lstsq import lstsq
from .manipulate_dimension import (
    append_dimension,
    insert_dimension,
    permute_dimensions,
    remove_dimension,
    rename_dimension,
)
from .multiply import multiply
from .one_hot import one_hot
from .ones_like import ones_like, ones_like_block
from .random_like import random_uniform_like, random_uniform_like_block
from .pow import pow
from .reduce_over_samples import (
    mean_over_samples,
    mean_over_samples_block,
    std_over_samples,
    std_over_samples_block,
    sum_over_samples,
    sum_over_samples_block,
    var_over_samples,
    var_over_samples_block,
)
from .remove_gradients import remove_gradients
from .slice import slice, slice_block
from .solve import solve
from .split import split, split_block
from .to import block_to, to
from .subtract import subtract
from .unique_metadata import unique_metadata, unique_metadata_block
from .checks import checks_enabled, unsafe_disable_checks, unsafe_enable_checks
from .zeros_like import zeros_like, zeros_like_block
from .sort import sort, sort_block

__all__ = [
    "NotEqualError",
    "abs",
    "add",
    "allclose",
    "allclose_block",
    "allclose_block_raise",
    "allclose_raise",
    "append_dimension",
    "block_from_array",
    "block_to",
    "checks_enabled",
    "divide",
    "dot",
    "drop_blocks",
    "empty_like",
    "empty_like_block",
    "equal",
    "equal_block",
    "equal_block_raise",
    "equal_metadata",
    "equal_metadata_block",
    "equal_metadata_block_raise",
    "equal_metadata_raise",
    "equal_raise",
    "insert_dimension",
    "join",
    "lstsq",
    "mean_over_samples",
    "mean_over_samples_block",
    "multiply",
    "one_hot",
    "ones_like",
    "ones_like_block",
    "permute_dimensions",
    "pow",
    "random_uniform_like",
    "random_uniform_like_block",
    "remove_gradients",
    "remove_dimension",
    "rename_dimension",
    "slice",
    "slice_block",
    "solve",
    "split",
    "split_block",
    "sort",
    "sort_block",
    "std_over_samples",
    "std_over_samples_block",
    "subtract",
    "sum_over_samples",
    "sum_over_samples_block",
    "to",
    "unique_metadata",
    "unique_metadata_block",
    "unsafe_disable_checks",
    "unsafe_enable_checks",
    "var_over_samples",
    "var_over_samples_block",
    "zeros_like",
    "zeros_like_block",
]
