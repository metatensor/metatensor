import importlib.metadata


__version__ = importlib.metadata.version("metatensor-operations")


from ._abs import abs  # noqa: F401
from ._add import add  # noqa: F401
from ._allclose import (  # noqa: F401
    allclose,
    allclose_block,
    allclose_block_raise,
    allclose_raise,
)
from ._block_from_array import block_from_array  # noqa: F401
from ._checks import (  # noqa: F401
    checks_enabled,
    unsafe_disable_checks,
    unsafe_enable_checks,
)
from ._detach import detach, detach_block  # noqa: F401
from ._divide import divide  # noqa: F401
from ._dot import dot  # noqa: F401
from ._drop_blocks import drop_blocks  # noqa: F401
from ._empty_like import empty_like, empty_like_block  # noqa: F401
from ._equal import equal, equal_block, equal_block_raise, equal_raise  # noqa: F401
from ._equal_metadata import (  # noqa: F401
    equal_metadata,
    equal_metadata_block,
    equal_metadata_block_raise,
    equal_metadata_raise,
)
from ._filter_blocks import filter_blocks  # noqa: F401
from ._is_contiguous import (  # noqa: F401
    is_contiguous,
    is_contiguous_block,
)
from ._join import join  # noqa: F401
from ._lstsq import lstsq  # noqa: F401
from ._make_contiguous import (  # noqa: F401
    make_contiguous,
    make_contiguous_block,
)
from ._manipulate_dimension import (  # noqa: F401
    append_dimension,
    insert_dimension,
    permute_dimensions,
    remove_dimension,
    rename_dimension,
)
from ._multiply import multiply  # noqa: F401
from ._one_hot import one_hot  # noqa: F401
from ._ones_like import ones_like, ones_like_block  # noqa: F401
from ._pow import pow  # noqa: F401
from ._random_like import random_uniform_like, random_uniform_like_block  # noqa: F401
from ._reduce_over_samples import (  # noqa: F401
    mean_over_samples,
    mean_over_samples_block,
    std_over_samples,
    std_over_samples_block,
    sum_over_samples,
    sum_over_samples_block,
    var_over_samples,
    var_over_samples_block,
)
from ._remove_gradients import remove_gradients, remove_gradients_block  # noqa: F401
from ._requires_grad import requires_grad, requires_grad_block  # noqa: F401
from ._slice import slice, slice_block  # noqa: F401
from ._solve import solve  # noqa: F401
from ._sort import sort, sort_block  # noqa: F401
from ._split import split, split_block  # noqa: F401
from ._subtract import subtract  # noqa: F401
from ._unique_metadata import unique_metadata, unique_metadata_block  # noqa: F401
from ._utils import NotEqualError  # noqa: F401
from ._zeros_like import zeros_like, zeros_like_block  # noqa: F401
