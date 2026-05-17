import mmap
import pathlib
from typing import Union

import numpy as np

from .._block import TensorBlock
from .._c_api import mts_create_file_array_callback_t
from .._c_lib import _get_library
from .._data._array import create_mts_array
from .._status import catch_exceptions
from .._tensor import TensorMap
from ._block import _dlpack_dtype_to_numpy


def _make_numpy_mmap_callback(mm: mmap.mmap):
    """
    Build a `mts_create_file_array_callback_t` that materialises each array
    as a numpy view into the given mmap.

    The mmap object is captured by the callback closure AND by every returned
    numpy array (via `np.frombuffer`'s `.base` chain), so it stays alive as
    long as any of the arrays do.


    same file to parse NPY headers. We therefore have *two* VA mappings of
    the same inode active during the load. The OS shares physical pages
    (both mappings are read-only of the same file) so the actual memory
    cost is one set of pages, not two. Removing the duplicate would require
    threading the C-side mmap base pointer through ``user_data`` to the
    callback -- the C API parameter is already ``void *`` for this kind of
    thing; deferred to a follow-up that touches the C surface.
    """

    @catch_exceptions
    def callback(_user_data, shape_ptr, shape_count, dtype, file_offset, array_out):
        shape = [shape_ptr[i] for i in range(shape_count)]
        np_dtype = _dlpack_dtype_to_numpy(dtype)
        # Audit #7: np.prod returns 1 for an empty list, so the explicit
        # ``if shape else 1`` guard is redundant.
        nelems = int(np.prod(shape, dtype=np.int64))
        if nelems == 0:
            data = np.empty(shape, dtype=np_dtype)
        else:
            data = np.frombuffer(
                mm, dtype=np_dtype, count=nelems, offset=file_offset
            ).reshape(shape)
            data.setflags(write=False)
        array_out[0] = create_mts_array(data)

    return callback


def load_mmap(path: Union[str, pathlib.Path]) -> TensorMap:
    """
    Load a previously saved :py:class:`TensorMap` from the given path using
    memory mapping.

    Numeric arrays are returned as read-only ``numpy`` views directly into
    the memory-mapped file (no copy). Labels are loaded normally.

    The input file must use the ``STORED`` (uncompressed) ZIP format that
    ``save`` produces, and numeric arrays must use native byte order.

    :param path: path of the file to load
    """
    if isinstance(path, pathlib.Path):
        path = str(path)
    # Audit #6: mmap.mmap keeps its own reference to the fd, so the with
    # block can close the original fd as soon as the mapping is set up.
    with open(path, "rb") as fd:
        mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)

    lib = _get_library()
    callback = _make_numpy_mmap_callback(mm)
    encoded = path.encode("utf8")
    ptr = lib.mts_tensormap_load_mmap(
        encoded,
        mts_create_file_array_callback_t(callback),
        None,
    )

    return TensorMap._from_ptr(ptr)


def load_block_mmap(path: Union[str, pathlib.Path]) -> TensorBlock:
    """
    Load a previously saved :py:class:`TensorBlock` from the given path using
    memory mapping. See :py:func:`load_mmap` for semantics.

    :param path: path of the file to load
    """
    if isinstance(path, pathlib.Path):
        path = str(path)
    # Audit #6: mmap.mmap keeps its own reference to the fd, so the with
    # block can close the original fd as soon as the mapping is set up.
    with open(path, "rb") as fd:
        mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)

    lib = _get_library()
    callback = _make_numpy_mmap_callback(mm)
    encoded = path.encode("utf8")
    ptr = lib.mts_block_load_mmap(
        encoded,
        mts_create_file_array_callback_t(callback),
        None,
    )

    return TensorBlock._from_ptr(ptr, parent=None)
