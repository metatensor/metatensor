import ctypes
import pathlib
import warnings
from typing import BinaryIO, Union

import numpy as np

from .._c_api import mts_labels_t
from .._c_lib import _get_library
from ..labels import Labels
from ._utils import _save_buffer_raw


def load_labels(file: Union[str, pathlib.Path, BinaryIO]) -> Labels:
    """
    Load previously saved :py:class:`Labels` from the given file.

    :param file: file to load: this can be a string, a :py:class:`pathlib.Path`
        containing the path to the file to load, or a file-like object opened in binary
        mode.
    """
    if isinstance(file, (str, pathlib.Path)):
        lib = _get_library()

        if isinstance(file, str):
            path = file.encode("utf8")
        elif isinstance(path, pathlib.Path):
            path = bytes(path)

        labels = mts_labels_t()
        lib.mts_labels_load(path, labels)
        return Labels._from_mts_labels_t(labels)

    else:
        # assume we have a file-like object
        buffer = file.read()
        assert isinstance(buffer, bytes)

        return load_labels_buffer(buffer)


def load_labels_buffer(buffer: Union[bytes, bytearray, memoryview]) -> Labels:
    """
    Load previously saved :py:class:`Labels` from an in-memory buffer.

    :param buffer: in-memory buffer containing saved :py:class:`Labels`
    """
    lib = _get_library()

    if isinstance(buffer, bytearray):
        char_array = ctypes.c_char * len(buffer)
        buffer = char_array.from_buffer(buffer)
    elif isinstance(buffer, memoryview):
        char_array = ctypes.c_char * len(buffer)
        # FIXME: we would prefer not to make a copy here, but ctypes does not support
        # passing a memory view to C, even if it is contiguous.
        # https://github.com/python/cpython/issues/60190
        buffer = char_array.from_buffer_copy(buffer)

    labels = mts_labels_t()
    lib.mts_labels_load_buffer(buffer, len(buffer), labels)

    return Labels._from_mts_labels_t(labels)


def _save_labels(
    file: Union[str, pathlib.Path, BinaryIO],
    labels: Labels,
):
    """
    Save :py:class:`Labels` to the given file.

    :param file: where to save the data. This can be a string, :py:class:`pathlib.Path`
        containing the path to the file to load, or a file-like object that should be
        opened in binary mode.
    :param labels: Labels to save
    """
    assert isinstance(labels, Labels)

    lib = _get_library()
    if isinstance(file, str):
        if not file.endswith(".npy"):
            file += ".npz"
            warnings.warn(
                message=f"adding '.npy' extension, the file will be saved at '{file}'",
                stacklevel=1,
            )
        path = file.encode("utf8")
        lib.mts_labels_save(path, labels._labels)
    elif isinstance(file, pathlib.Path):
        if not file.name.endswith(".npy"):
            file = file.with_name(file.name + ".npy")
            warnings.warn(
                message="adding '.npy' extension,"
                f" the file will be saved at '{file.name}'",
                stacklevel=1,
            )
        path = bytes(file)
        lib.mts_labels_save(path, labels._labels)
    else:
        # assume we have a file-like object
        buffer = _save_labels_buffer_raw(labels)
        file.write(buffer.raw)


def _save_labels_buffer_raw(labels: Labels) -> ctypes.Array:
    """
    Save Labels to an in-memory buffer, returning the data as a ctypes array of
    ``ctypes.c_char``.
    """
    lib = _get_library()

    return _save_buffer_raw(lib.mts_labels_save_buffer, labels._labels)


def _labels_from_npz(data):
    names = data.dtype.names
    return Labels(names=names, values=data.view(dtype=np.int32).reshape(-1, len(names)))


def _labels_to_npz(labels):
    dtype = [(name, np.int32) for name in labels.names]
    return labels.values.view(dtype=dtype).reshape((labels.values.shape[0],))
