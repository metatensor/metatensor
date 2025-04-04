import io
import pathlib
from typing import BinaryIO, Union

from ..block import TensorBlock
from ..labels import Labels
from ..tensor import TensorMap
from ._block import (  # noqa: F401
    _save_block,
    _save_block_buffer_raw,
    create_numpy_array,
    create_torch_array,
    load_block,
    load_block_buffer,
    load_block_buffer_custom_array,
    load_block_custom_array,
)
from ._labels import (  # noqa: F401
    _save_labels,
    _save_labels_buffer_raw,
    load_labels,
    load_labels_buffer,
)
from ._tensor import (  # noqa: F401
    _save_tensor,
    _save_tensor_buffer_raw,
    load,
    load_buffer,
    load_buffer_custom_array,
    load_custom_array,
)


def save(
    file: Union[str, pathlib.Path, BinaryIO],
    data: Union[TensorMap, TensorBlock, Labels],
    use_numpy=False,
):
    """
    Save the given data (one of :py:class:`TensorMap`, :py:class:`TensorBlock`, or
    :py:class:`Labels`) to the given ``file``.

    :py:class:`TensorMap` are serialized using numpy's ``NPZ`` format, i.e. a ZIP file
    without compression (storage method is ``STORED``), where each file is stored as a
    ``.npy`` array. See the C API documentation for more information on the format.

    The recomended file extension when saving data is ``.mts``, to prevent confusion
    with generic ``.npz`` files.

    :param file: where to save the data. This can be a string, :py:class:`pathlib.Path`
        containing the path to the file to load, or a file-like object that should be
        opened in binary mode.
    :param data: data to serialize and save
    :param use_numpy: should we use numpy or the native serializer implementation? Numpy
        should be able to process more dtypes than the native implementation, which is
        limited to float64, but the native implementation is usually faster than going
        through numpy. This is ignored when saving :py:class:`Labels`.
    """
    if isinstance(data, Labels):
        return _save_labels(file=file, labels=data)
    elif isinstance(data, TensorBlock):
        return _save_block(file=file, block=data, use_numpy=use_numpy)
    elif isinstance(data, TensorMap):
        return _save_tensor(file=file, tensor=data, use_numpy=use_numpy)
    else:
        raise TypeError(
            "`data` must be one of 'Labels', 'TensorBlock' or 'TensorMap', "
            f"not {type(data)}"
        )


def save_buffer(
    data: Union[TensorMap, TensorBlock, Labels],
    use_numpy=False,
) -> memoryview:
    """
    Save the given data (one of :py:class:`TensorMap`, :py:class:`TensorBlock`, or
    :py:class:`Labels`) to an in-memory buffer.

    :param data: data to serialize and save
    :param use_numpy: should we use numpy or the native serializer implementation?
    """
    if isinstance(data, Labels):
        return memoryview(_save_labels_buffer_raw(labels=data))
    elif isinstance(data, TensorBlock):
        if use_numpy:
            file = io.BytesIO()
            save(file, data=data, use_numpy=use_numpy)
            return file.getbuffer()
        else:
            return memoryview(_save_block_buffer_raw(block=data))
    elif isinstance(data, TensorMap):
        if use_numpy:
            file = io.BytesIO()
            save(file, data=data, use_numpy=use_numpy)
            return file.getbuffer()
        else:
            return memoryview(_save_tensor_buffer_raw(tensor=data))
    else:
        raise TypeError(
            "`data` must be one of 'Labels', 'TensorBlock' or 'TensorMap', "
            f"not {type(data)}"
        )
