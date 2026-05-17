import ctypes
import pathlib
from typing import Any, Optional, Union

from .._block import TensorBlock
from .._c_api import mts_create_array_callback_t, mts_labels_t
from .._c_lib import _get_library
from .._labels import Labels
from .._tensor import TensorMap
from ._block import create_numpy_array


def _labels_arg(labels: Optional[Labels]) -> Any:
    """Map ``labels`` to an ``mts_labels_t *`` for the C API. ``None``
    becomes a NULL pointer, which the C core interprets as 'select all'
    on this dimension. Audit #17: annotation is ``Any`` because
    ``ctypes.POINTER(...)`` is a callable, not a real type."""
    if labels is None:
        return ctypes.POINTER(mts_labels_t)()
    return labels._as_mts_labels_t()


def load_partial(
    path: Union[str, pathlib.Path],
    keys: Optional[Labels] = None,
    samples: Optional[Labels] = None,
    properties: Optional[Labels] = None,
) -> TensorMap:
    """
    Load a previously saved :py:class:`TensorMap` from ``path``, selecting
    only a subset of blocks (``keys``), rows (``samples``), and columns
    (``properties``).

    Each of ``keys`` / ``samples`` / ``properties`` is optional: passing
    ``None`` (the default) selects everything on that dimension. When
    provided, selection uses :py:meth:`Labels.select` semantics.

    The returned tensor owns its arrays (the underlying mmap is dropped
    before this function returns); rows / columns are byte-copied out of
    the memory-mapped file by the C core.

    The input file must use the ``STORED`` (uncompressed) ZIP format that
    :py:func:`save` produces, and numeric arrays must use native byte
    order.

    :param path: path of the file to load
    :param keys: optional key selector
    :param samples: optional sample selector
    :param properties: optional property selector
    """
    if isinstance(path, pathlib.Path):
        path = str(path)
    encoded = path.encode("utf8")

    lib = _get_library()
    ptr = lib.mts_tensormap_load_partial(
        encoded,
        _labels_arg(keys),
        _labels_arg(samples),
        _labels_arg(properties),
        mts_create_array_callback_t(create_numpy_array),
    )
    return TensorMap._from_ptr(ptr)


def load_block_partial(
    path: Union[str, pathlib.Path],
    samples: Optional[Labels] = None,
    properties: Optional[Labels] = None,
) -> TensorBlock:
    """
    Load a previously saved :py:class:`TensorBlock` from ``path``,
    selecting only a subset of samples and properties. See
    :py:func:`load_partial` for semantics.

    :param path: path of the file to load
    :param samples: optional sample selector
    :param properties: optional property selector
    """
    if isinstance(path, pathlib.Path):
        path = str(path)
    encoded = path.encode("utf8")

    lib = _get_library()
    ptr = lib.mts_block_load_partial(
        encoded,
        _labels_arg(samples),
        _labels_arg(properties),
        mts_create_array_callback_t(create_numpy_array),
    )
    return TensorBlock._from_ptr(ptr, parent=None)
