import ctypes
import io
import pathlib
import warnings
from typing import BinaryIO, Union

import numpy as np

from .._c_api import mts_create_array_callback_t
from .._c_lib import _get_library
from ..tensor import TensorMap
from ._block import (
    CreateArrayCallback,
    _block_to_dict,
    _single_block_from_mts,
    create_numpy_array,
)
from ._labels import _labels_from_mts, _labels_to_mts
from ._utils import _save_buffer_raw


def load(file: Union[str, pathlib.Path, BinaryIO], use_numpy=False) -> TensorMap:
    """
    Load a previously saved :py:class:`TensorMap` from the given file.

    :py:class:`TensorMap` are serialized using numpy's NPZ format, i.e. a ZIP file
    without compression (storage method is ``STORED``), where each file is stored as a
    ``.npy`` array. See the C API documentation for more information on the format.

    :param file: file to load: this can be a string, a :py:class:`pathlib.Path`
        containing the path to the file to load, or a file-like object that should be
        opened in binary mode.
    :param use_numpy: should we use numpy or the native implementation? Numpy should be
        able to process more dtypes than the native implementation, which is limited to
        float64, but the native implementation is usually faster than going through
        numpy.
    """
    if use_numpy:
        return _tensor_from_mts(file)
    else:
        if isinstance(file, (str, pathlib.Path)):
            return load_custom_array(file, create_numpy_array)
        else:
            # assume we have a file-like object
            buffer = file.read()
            assert isinstance(buffer, bytes)

            return load_buffer_custom_array(buffer, create_numpy_array)


def load_buffer(
    buffer: Union[bytes, bytearray, memoryview], use_numpy=False
) -> TensorMap:
    """
    Load a previously saved :py:class:`TensorMap` from an in-memory buffer.

    :param buffer: In-memory buffer containing a serialized ``TensorMap``
    :param use_numpy: should we use numpy or the native implementation?
    """
    if use_numpy:
        return _tensor_from_mts(io.BytesIO(buffer))
    else:
        return load_buffer_custom_array(buffer, create_numpy_array)


def load_custom_array(
    path: Union[str, pathlib.Path],
    create_array: "CreateArrayCallback",
) -> TensorMap:
    """
    Load a previously saved :py:class:`TensorMap` from the given path using a custom
    array creation callback.

    This is an advanced functionality, which should not be needed by most users.

    This function allows to specify the kind of array to use when loading the data
    through the ``create_array`` callback. This callback should take three arguments: a
    pointer to the shape, the number of elements in the shape, and a pointer to the
    ``mts_array_t`` to be filled.

    :py:func:`metatensor.io.create_numpy_array` and
    :py:func:`metatensor.io.create_torch_array` can be used to load data into numpy
    and torch arrays respectively.

    :param path: path of the file to load
    :param create_array: callback used to create arrays as needed
    """

    lib = _get_library()

    if isinstance(path, str):
        path = path.encode("utf8")
    elif isinstance(path, pathlib.Path):
        path = bytes(path)

    ptr = lib.mts_tensormap_load(path, mts_create_array_callback_t(create_array))

    return TensorMap._from_ptr(ptr)


def load_buffer_custom_array(
    buffer: Union[bytes, bytearray, memoryview],
    create_array: "CreateArrayCallback",
) -> TensorMap:
    """
    Load a previously saved :py:class:`TensorMap` from the given buffer using a custom
    array creation callback.

    This is an advanced functionality, which should not be needed by most users.

    This function allows to specify the kind of array to use when loading the data
    through the ``create_array`` callback. This callback should take three arguments: a
    pointer to the shape, the number of elements in the shape, and a pointer to the
    ``mts_array_t`` to be filled.

    :py:func:`metatensor.io.create_numpy_array` and
    :py:func:`metatensor.io.create_torch_array` can be used to load data into numpy
    and torch arrays respectively.

    :param buffer: in-memory buffer containing a saved :py:class:`TensorMap`
    :param create_array: callback used to create arrays as needed
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

    ptr = lib.mts_tensormap_load_buffer(
        buffer,
        len(buffer),
        mts_create_array_callback_t(create_array),
    )

    return TensorMap._from_ptr(ptr)


def _save_tensor(
    file: Union[str, pathlib.Path, BinaryIO],
    tensor: TensorMap,
    use_numpy=False,
):
    assert isinstance(tensor, TensorMap)

    if use_numpy:
        all_entries = _tensor_to_dict(tensor)
        if not hasattr(file, "write"):
            # prevent numpy from adding a .npz extension by opening the file ourself
            with open(file, "wb") as fd:
                np.savez(fd, **all_entries)
        else:
            np.savez(file, **all_entries)
    else:
        lib = _get_library()
        if isinstance(file, str):
            if not file.endswith(".mts"):
                file += ".mts"
                warnings.warn(
                    message="adding '.mts' extension,"
                    f" the file will be saved at '{file}'",
                    stacklevel=1,
                )
            path = file.encode("utf8")
            lib.mts_tensormap_save(path, tensor._ptr)
        elif isinstance(file, pathlib.Path):
            if not file.name.endswith(".mts"):
                file = file.with_name(file.name + ".mts")
                warnings.warn(
                    message="adding '.mts' extension,"
                    f" the file will be saved at '{file.name}'",
                    stacklevel=1,
                )
            path = bytes(file)
            lib.mts_tensormap_save(path, tensor._ptr)
        else:
            # assume we have a file-like object
            buffer = _save_tensor_buffer_raw(tensor)
            file.write(buffer.raw)


def _save_tensor_buffer_raw(tensor: TensorMap) -> ctypes.Array:
    """
    Save a TensorMap to an in-memory buffer, returning the data as a ctypes array of
    ``ctypes.c_char``.
    """
    lib = _get_library()

    return _save_buffer_raw(lib.mts_tensormap_save_buffer, tensor._ptr)


def _tensor_to_dict(tensor_map):
    result = {
        "keys": _labels_to_mts(tensor_map.keys),
    }

    for block_i, block in enumerate(tensor_map.blocks()):
        prefix = f"blocks/{block_i}/"
        result.update(_block_to_dict(block, prefix, is_gradient=False))

    return result


def _tensor_from_mts(file):
    dictionary = np.load(file)

    keys = _labels_from_mts(dictionary["keys"])
    blocks = []

    for block_i in range(len(keys)):
        prefix = f"blocks/{block_i}/"
        properties = _labels_from_mts(dictionary[f"{prefix}properties"])

        block = _single_block_from_mts(prefix, dictionary, properties)
        blocks.append(block)

    return TensorMap(keys, blocks)
