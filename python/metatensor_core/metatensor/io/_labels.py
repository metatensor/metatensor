import ctypes
import pathlib
import warnings
from typing import BinaryIO, Union

import numpy as np

from .._c_api import mts_create_array_callback_t
from .._c_lib import _get_library
from .._labels import Labels
from .._status import check_pointer
from ._utils import CreateArrayCallback, _save_buffer_raw, create_numpy_array


def load_labels(file: Union[str, pathlib.Path, BinaryIO]) -> Labels:
    """
    Load previously saved :py:class:`Labels` from the given file.

    :param file: file to load: this can be a string, a :py:class:`pathlib.Path`
        containing the path to the file to load, or a file-like object opened in binary
        mode.
    """
    if isinstance(file, (str, pathlib.Path)):
        return load_labels_custom_array(file, create_numpy_array)
    else:
        # assume we have a file-like object
        buffer = file.read()
        assert isinstance(buffer, bytes)

        return load_labels_buffer_custom_array(buffer, create_numpy_array)


def load_labels_buffer(buffer: Union[bytes, bytearray, memoryview]) -> Labels:
    """
    Load previously saved :py:class:`Labels` from an in-memory buffer.

    :param buffer: in-memory buffer containing saved :py:class:`Labels`
    """
    return load_labels_buffer_custom_array(buffer, create_numpy_array)


def load_labels_custom_array(
    path: Union[str, pathlib.Path],
    create_array: "CreateArrayCallback",
) -> Labels:
    """
    Load previously saved :py:class:`Labels` from the given path using a custom array
    creation callback.

    This is an advanced functionality, which should not be needed by most users.

    This function allows to specify the kind of array to use when loading the labels
    values through the ``create_array`` callback. This callback should take four
    arguments: a pointer to the shape, the number of elements in the shape, the
    ``DLDataType`` of the array, and a pointer to the ``mts_array_t`` to be filled.

    :py:func:`metatensor.io.create_numpy_array` and
    :py:func:`metatensor.io.create_torch_array` can be used to load data into numpy and
    torch arrays respectively.

    :param path: path of the file to load
    :param create_array: callback used to create arrays as needed
    """
    lib = _get_library()

    if isinstance(path, str):
        path = path.encode("utf8")
    elif isinstance(path, pathlib.Path):
        path = bytes(path)

    ptr = lib.mts_labels_load_custom_array(
        path, mts_create_array_callback_t(create_array)
    )
    check_pointer(ptr)
    return Labels.unsafe_from_ptr(ptr)


def load_labels_buffer_custom_array(
    buffer: Union[bytes, bytearray, memoryview],
    create_array: "CreateArrayCallback",
) -> Labels:
    """
    Load previously saved :py:class:`Labels` from the given buffer using a custom array
    creation callback.

    This is an advanced functionality, which should not be needed by most users.

    This function allows to specify the kind of array to use when loading the labels
    values through the ``create_array`` callback. This callback should take four
    arguments: a pointer to the shape, the number of elements in the shape, the
    ``DLDataType`` of the array, and a pointer to the ``mts_array_t`` to be filled.

    :py:func:`metatensor.io.create_numpy_array` and
    :py:func:`metatensor.io.create_torch_array` can be used to load data into numpy
    and torch arrays respectively.

    :param buffer: in-memory buffer containing saved :py:class:`Labels`
    :param create_array: callback used to create arrays as needed
    """
    lib = _get_library()

    array = np.frombuffer(buffer, dtype=np.uint8)
    ptr = lib.mts_labels_load_buffer_custom_array(
        array.ctypes.data_as(ctypes.c_char_p),
        array.nbytes,
        mts_create_array_callback_t(create_array),
    )
    check_pointer(ptr)

    return Labels.unsafe_from_ptr(ptr)


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
        if not file.endswith(".mts"):
            file += ".mts"
            warnings.warn(
                message=f"adding '.mts' extension, the file will be saved at '{file}'",
                stacklevel=1,
            )
        path = file.encode("utf8")
        lib.mts_labels_save(path, labels._ptr)
    elif isinstance(file, pathlib.Path):
        if not file.name.endswith(".mts"):
            file = file.with_name(file.name + ".mts")
            warnings.warn(
                message="adding '.mts' extension,"
                f" the file will be saved at '{file.name}'",
                stacklevel=1,
            )
        path = bytes(file)
        lib.mts_labels_save(path, labels._ptr)
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

    return _save_buffer_raw(lib.mts_labels_save_buffer, labels._ptr)


def _labels_from_mts(data):
    names = data.dtype.names
    return Labels(names=names, values=data.view(dtype=np.int32).reshape(-1, len(names)))


def _labels_to_mts(labels):
    dtype = [(name, np.int32) for name in labels.names]
    values = np.asarray(labels.values)
    return values.view(dtype=dtype).reshape((values.shape[0],)).copy()
