import os
import pathlib
import warnings
from typing import BinaryIO, Union

import torch


if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX", "0") != "0":
    from ._documentation import (
        Labels,
        TensorBlock,
        TensorMap,
    )
else:
    Labels = torch.classes.metatensor.Labels
    TensorBlock = torch.classes.metatensor.TensorBlock
    TensorMap = torch.classes.metatensor.TensorMap


# the functions in this file are written in a relatively strange way to enable them
# taking `Path` and `BinaryIO` arguments when running in Python mode, while they only
# support str in TorchScript mode


def load(file: str, mmap: bool = False) -> TensorMap:
    """
    Load a previously saved :py:class:`TensorMap` from the given ``file``.

    :py:class:`TensorMap` are serialized using the ``.mts`` format, i.e. a ZIP file
    without compression (storage method is ``STORED``), where each file is stored as a
    ``.npy`` array. See the C API documentation for more information on the format.

    :param file: path of the file to load, or already opened file.

        .. warning::

            When using this function in TorchScript mode, only ``str`` arguments are
            supported.
    :param mmap: if ``True``, return memory-mapped tensors when the stored payloads
        are aligned for their dtypes. Unaligned payloads are copied into aligned
        tensors. Requires ``file`` to be a real filesystem path; file-like objects
        are rejected.
    """
    if torch.jit.is_scripting():
        assert isinstance(file, str)
        return torch.ops.metatensor.load(file=file, mmap=mmap)
    else:
        if isinstance(file, str):
            return torch.ops.metatensor.load(file=file, mmap=mmap)
        elif isinstance(file, pathlib.Path):
            return torch.ops.metatensor.load(file=str(file.resolve()), mmap=mmap)
        else:
            if mmap:
                raise TypeError(
                    "mmap=True requires a filesystem path; file-like objects "
                    "are not supported because mmap is a file-system operation"
                )
            # assume a file-like object
            buffer = file.read()
            assert isinstance(buffer, bytes)

            with warnings.catch_warnings():
                # ignore warning about buffer beeing non-writeable
                warnings.simplefilter("ignore")
                buffer = torch.frombuffer(buffer, dtype=torch.uint8)

            return torch.ops.metatensor.load_buffer(buffer=buffer)


# modify the annotations in a way such that the TorchScript compiler does not see these,
# but sphinx does for documentation.
load.__annotations__["file"] = Union[str, pathlib.Path, BinaryIO]


def load_block(file: str, mmap: bool = False) -> TensorBlock:
    """
    Load previously saved :py:class:`TensorBlock` from the given ``file``.

    :param path: path of the file to load, or already opened file.

        .. warning::

            When using this function in TorchScript mode, only ``str`` arguments are
            supported.
    :param mmap: if ``True``, return memory-mapped tensors when the stored payloads
        are aligned for their dtypes. Unaligned payloads are copied into aligned
        tensors. Requires ``file`` to be a real filesystem path.
    """
    if torch.jit.is_scripting():
        assert isinstance(file, str)
        return torch.ops.metatensor.load_block(file=file, mmap=mmap)
    else:
        if isinstance(file, str):
            return torch.ops.metatensor.load_block(file=file, mmap=mmap)
        elif isinstance(file, pathlib.Path):
            return torch.ops.metatensor.load_block(file=str(file.resolve()), mmap=mmap)
        else:
            if mmap:
                raise TypeError(
                    "mmap=True requires a filesystem path; file-like objects "
                    "are not supported because mmap is a file-system operation"
                )
            # assume a file-like object
            buffer = file.read()
            assert isinstance(buffer, bytes)

            with warnings.catch_warnings():
                # ignore warning about buffer beeing non-writeable
                warnings.simplefilter("ignore")
                buffer = torch.frombuffer(buffer, dtype=torch.uint8)

            return torch.ops.metatensor.load_block_buffer(buffer=buffer)


load_block.__annotations__["file"] = Union[str, pathlib.Path, BinaryIO]


def load_labels(file: str, mmap: bool = False) -> Labels:
    """
    Load previously saved :py:class:`Labels` from the given ``file``.

    :param file: path of the file to load, or already opened file.

        .. warning::

            When using this function in TorchScript mode, only ``str`` arguments are
            supported.
    :param mmap: if ``True``, return Labels whose entry data is memory-mapped when
        the stored payload is aligned for ``int32``. Unaligned payloads are copied
        into aligned tensors. Requires ``file`` to be a real filesystem path.
    """
    if torch.jit.is_scripting():
        assert isinstance(file, str)
        return torch.ops.metatensor.load_labels(file=file, mmap=mmap)
    else:
        if isinstance(file, str):
            return torch.ops.metatensor.load_labels(file=file, mmap=mmap)
        elif isinstance(file, pathlib.Path):
            return torch.ops.metatensor.load_labels(file=str(file.resolve()), mmap=mmap)
        else:
            if mmap:
                raise TypeError(
                    "mmap=True requires a filesystem path; file-like objects "
                    "are not supported because mmap is a file-system operation"
                )
            # assume a file-like object
            buffer = file.read()
            assert isinstance(buffer, bytes)

            with warnings.catch_warnings():
                # ignore warning about buffer beeing non-writeable
                warnings.simplefilter("ignore")
                buffer = torch.frombuffer(buffer, dtype=torch.uint8)

            return torch.ops.metatensor.load_labels_buffer(buffer=buffer)


load_labels.__annotations__["file"] = Union[str, pathlib.Path, BinaryIO]


def save(file: str, data: Union[TensorMap, TensorBlock, Labels]):
    """
    Save the given data (either :py:class:`TensorMap`, :py:class:`TensorBlock`, or
    :py:class:`Labels`) to the given ``file``.

    If the file already exists, it is overwritten. The recommended file extension when
    saving data is ``.mts``, to prevent confusion with generic ``.npz`` files.

    :param file: path of the file where to save the data, or already opened file.

        .. warning::

            When using this function in TorchScript mode, only ``str`` arguments are
            supported.

    :param data: data to serialize and save
    """
    if torch.jit.is_scripting():
        assert isinstance(file, str)
        return torch.ops.metatensor.save(file=file, data=data)
    else:
        if isinstance(file, str):
            return torch.ops.metatensor.save(file=file, data=data)
        elif isinstance(file, pathlib.Path):
            return torch.ops.metatensor.save(file=str(file.resolve()), data=data)
        else:
            # assume a file-like object
            buffer = torch.ops.metatensor.save_buffer(data=data)
            assert isinstance(buffer, torch.Tensor)
            file.write(buffer.numpy().tobytes())


save.__annotations__["file"] = Union[str, pathlib.Path, BinaryIO]
