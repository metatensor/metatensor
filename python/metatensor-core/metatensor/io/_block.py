# from __future__ import annotations

import ctypes
import io
import pathlib
import warnings
from typing import BinaryIO, Callable, Union

import numpy as np

from .._c_api import c_uintptr_t, mts_array_t, mts_create_array_callback_t
from .._c_lib import _get_library
from ..block import TensorBlock
from ..data.array import ArrayWrapper, _is_numpy_array, _is_torch_array
from ..utils import catch_exceptions
from ._labels import _labels_from_npz, _labels_to_npz
from ._utils import _save_buffer_raw


# TODO: use a proper type alias when we drop support for Python <3.10; and remove the
# quotes around the type annotations using this.
# https://stackoverflow.com/a/73223518/4692076
CreateArrayCallback = Callable[
    [ctypes.POINTER(c_uintptr_t), c_uintptr_t, ctypes.POINTER(mts_array_t)], None
]


@catch_exceptions
def create_numpy_array(shape_ptr, shape_count, array):
    """
    Callback function that can be used with
    :py:func:`metatensor.io.load_custom_array` to load data in numpy arrays.
    """
    shape = []
    for i in range(shape_count):
        shape.append(shape_ptr[i])

    data = np.empty(shape, dtype=np.float64)
    wrapper = ArrayWrapper(data)
    array[0] = wrapper.into_mts_array()


@catch_exceptions
def create_torch_array(shape_ptr, shape_count, array):
    """
    Callback function that can be used with
    :py:func:`metatensor.io.load_custom_array` to load data in torch
    tensors. The resulting tensors are stored on CPU, and their dtype is
    ``torch.float64``.
    """
    import torch

    shape = []
    for i in range(shape_count):
        shape.append(shape_ptr[i])

    data = torch.empty(shape, dtype=torch.float64, device="cpu")
    wrapper = ArrayWrapper(data)
    array[0] = wrapper.into_mts_array()


def load_block(
    file: Union[str, pathlib.Path, BinaryIO], use_numpy=False
) -> TensorBlock:
    """
    Load a previously saved :py:class:`TensorBlock` from the given file.

    :param file: file to load: this can be a string, a :py:class:`pathlib.Path`
        containing the path to the file to load, or a file-like object that should be
        opened in binary mode.
    :param use_numpy: should we use numpy or the native implementation? Numpy should be
        able to process more dtypes than the native implementation, which is limited to
        float64, but the native implementation is usually faster than going through
        numpy.
    """
    if use_numpy:
        return _block_from_npz(file)
    else:
        if isinstance(file, (str, pathlib.Path)):
            return load_block_custom_array(file, create_numpy_array)
        else:
            # assume we have a file-like object
            buffer = file.read()
            assert isinstance(buffer, bytes)

            return load_block_buffer_custom_array(buffer, create_numpy_array)


def load_block_buffer(
    buffer: Union[bytes, bytearray, memoryview], use_numpy=False
) -> TensorBlock:
    """
    Load a previously saved :py:class:`TensorBlock` from an in-memory buffer.

    :param buffer: In-memory buffer containing a serialized ``TensorBlock``
    :param use_numpy: should we use numpy or the native implementation?
    """
    if use_numpy:
        return _block_from_npz(io.BytesIO(buffer))
    else:
        return load_block_buffer_custom_array(buffer, create_numpy_array)
    pass


def load_block_custom_array(
    path: Union[str, pathlib.Path],
    create_array: "CreateArrayCallback",
) -> TensorBlock:
    """
    Load a previously saved :py:class:`TensorBlock` from the given path using a custom
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

    ptr = lib.mts_block_load(path, mts_create_array_callback_t(create_array))

    return TensorBlock._from_ptr(ptr, parent=None)


def load_block_buffer_custom_array(
    buffer: Union[bytes, bytearray, memoryview],
    create_array: "CreateArrayCallback",
) -> TensorBlock:
    """
    Load a previously saved :py:class:`TensorBlock` from the given buffer using a custom
    array creation callback.

    This is an advanced functionality, which should not be needed by most users.

    This function allows to specify the kind of array to use when loading the data
    through the ``create_array`` callback. This callback should take three arguments: a
    pointer to the shape, the number of elements in the shape, and a pointer to the
    ``mts_array_t`` to be filled.

    :py:func:`metatensor.io.create_numpy_array` and
    :py:func:`metatensor.io.create_torch_array` can be used to load data into numpy and
    torch arrays respectively.

    :param buffer: in-memory buffer containing a saved :py:class:`TensorBlock`
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

    ptr = lib.mts_block_load_buffer(
        buffer,
        len(buffer),
        mts_create_array_callback_t(create_array),
    )

    return TensorBlock._from_ptr(ptr, parent=None)


def _save_block(
    file: Union[str, pathlib.Path, BinaryIO],
    block: TensorBlock,
    use_numpy=False,
):
    assert isinstance(block, TensorBlock)

    if isinstance(file, (str, pathlib.Path)):
        if not file.endswith(".npz"):
            file += ".npz"
            warnings.warn(
                message=f"adding '.npz' extension, the file will be saved at '{file}'",
                stacklevel=1,
            )

    if use_numpy:
        all_entries = _block_to_dict(block, prefix="", is_gradient=False)
        np.savez(file, **all_entries)
    else:
        lib = _get_library()
        if isinstance(file, (str, pathlib.Path)):
            if isinstance(file, str):
                path = file.encode("utf8")
            elif isinstance(file, pathlib.Path):
                path = bytes(file)

            lib.mts_block_save(path, block._ptr)
        else:
            # assume we have a file-like object
            buffer = _save_block_buffer_raw(block)
            file.write(buffer.raw)


def _save_block_buffer_raw(block: TensorBlock) -> ctypes.Array:
    """
    Save a TensorBlock to an in-memory buffer, returning the data as a ctypes array of
    ``ctypes.c_char``.
    """
    lib = _get_library()

    return _save_buffer_raw(lib.mts_block_save_buffer, block._ptr)


def _block_to_dict(block, prefix, is_gradient):
    result = {}
    result[f"{prefix}values"] = _array_to_numpy(block.values)
    result[f"{prefix}samples"] = _labels_to_npz(block.samples)
    for i, component in enumerate(block.components):
        result[f"{prefix}components/{i}"] = _labels_to_npz(component)

    if not is_gradient:
        result[f"{prefix}properties"] = _labels_to_npz(block.properties)

    for parameter, gradient in block.gradients():
        result.update(
            _block_to_dict(
                gradient,
                f"{prefix}gradients/{parameter}/",
                is_gradient=True,
            )
        )

    return result


def _array_to_numpy(array):
    if _is_numpy_array(array):
        return array
    elif _is_torch_array(array):
        return array.cpu().numpy()
    else:
        raise ValueError("unknown array type passed to `metatensor.save`")


def _single_block_from_npz(prefix, dictionary, properties):
    values = dictionary[f"{prefix}values"]

    samples = _labels_from_npz(dictionary[f"{prefix}samples"])
    components = []
    for i in range(len(values.shape) - 2):
        components.append(_labels_from_npz(dictionary[f"{prefix}components/{i}"]))

    block = TensorBlock(values, samples, components, properties)

    parameters = set()
    gradient_prefix = f"{prefix}gradients/"
    for name in dictionary.keys():
        if name.startswith(gradient_prefix) and name.endswith("/values"):
            parameter = name[len(gradient_prefix) :]
            parameter = parameter.split("/")[0]
            parameters.add(parameter)

    for parameter in parameters:
        gradient = _single_block_from_npz(
            f"{prefix}gradients/{parameter}/",
            dictionary,
            properties,
        )
        block.add_gradient(parameter, gradient)

    return block


def _block_from_npz(file):
    dictionary = np.load(file)

    properties = _labels_from_npz(dictionary["properties"])
    return _single_block_from_npz("", dictionary, properties)