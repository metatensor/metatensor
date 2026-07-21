import ctypes
import pathlib
from pickle import PickleBuffer
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Union

import numpy as np
from ctypes_dlpack import (
    DLDevice,
    DLDeviceType,
    DLManagedTensorVersioned,
    DLPackVersion,
)

from . import _data
from ._block import TensorBlock
from ._c_api import (
    c_uintptr_t,
    mts_array_t,
    mts_block_t,
    mts_tensormap_t,
)
from ._c_lib import _get_library
from ._data import Device, DType
from ._labels import Labels, LabelsEntry
from ._status import check_pointer, check_status


class TensorMap:
    """
    A TensorMap is the main user-facing class of this library, and can store any kind of
    data used in atomistic machine learning similar to a Python :py:class:`dict`.

    A tensor map contains a list of :py:class:`TensorBlock`, each one associated with a
    key. Blocks can either be accessed one by one with the :py:func:`TensorMap.block`
    function, or by iterating over the tensor map itself:

    .. code-block:: python

        for block in tensor:
            ...

    The corresponding keys can be included in the loop by using the ``items()`` method
    of a :py:func:`TensorMap`:

    .. code-block:: python

        for key, block in tensor.items():
            ...

    A tensor map provides functions to move some of these keys to the samples or
    properties labels of the blocks, moving from a sparse representation of the data to
    a dense one.
    """

    def __init__(self, keys: Labels, blocks: Sequence[TensorBlock]):
        """
        :param keys: keys associated with each block
        :param blocks: set of blocks containing the actual data
        """
        if not isinstance(keys, Labels):
            raise TypeError(f"`keys` must be metatensor Labels, not {type(keys)}")

        blocks = list(blocks)
        for block in blocks:
            if not isinstance(block, TensorBlock):
                raise TypeError(
                    "`blocks` elements must be metatensor TensorBlock, "
                    f"not {type(block)}"
                )

        self._lib = _get_library()
        self._parent = None

        blocks_array_t = ctypes.POINTER(mts_block_t) * len(blocks)
        blocks_array = blocks_array_t(*[block.release() for block in blocks])

        self._ptr = self._lib.mts_tensormap(
            keys.as_mts_labels_t(), blocks_array, len(blocks)
        )
        check_pointer(self._ptr)

    @staticmethod
    def unsafe_from_ptr(tensor: ctypes.POINTER(mts_tensormap_t)):
        """
        Create a :py:class:`TensorMap` from a raw ``mts_tensormap_t`` pointer.

        The :py:class:`TensorMap` takes ownership of the pointer, and will
        release the corresponding memory when garbage-collected.
        """
        assert tensor, "mts_tensormap_t pointer is null"
        obj = TensorMap.__new__(TensorMap)
        obj._lib = _get_library()
        obj._ptr = tensor
        obj._parent = None
        return obj

    @staticmethod
    def unsafe_view_from_ptr(tensor: ctypes.POINTER(mts_tensormap_t), parent: Any):
        """
        Create a :py:class:`TensorMap` from a raw ``mts_tensormap_t`` pointer, keeping a
        reference to the ``parent`` to prevent garbage collection.

        The :py:class:`TensorMap` does **not** take ownership of the pointer, and will
        not release the memory.
        """
        assert parent is not None, (
            "please use TensorMap.unsafe_from_ptr to take ownership of a pointer"
        )

        obj = TensorMap.unsafe_from_ptr(tensor)
        obj._parent = parent
        return obj

    def as_mts_tensormap_t(self) -> ctypes.POINTER(mts_tensormap_t):
        """
        Get the underlying C pointer for this :py:class:`TensorMap`.

        This class still manages the tensor map memory after the call. Use
        :py:meth:`TensorMap.release` to take ownership of the pointer.
        """
        if not self._ptr:
            raise ValueError(
                "this TensorMap has been released and can no longer be used"
            )

        return self._ptr

    def release(self) -> ctypes.POINTER(mts_tensormap_t):
        """
        Release the underlying C pointer of this :py:class:`TensorMap`.

        This class is no longer managing the tensor map memory after the call,
        the user is expected to re-create a :py:class:`TensorMap` with
        :py:meth:`TensorMap.unsafe_from_ptr`, or pass the pointer to a C function
        that will call ``mts_tensormap_free``.
        """
        if self._parent is not None:
            raise RuntimeError(
                "can not release this TensorMap, it is already a view inside "
                "another TensorMap or TensorBlock"
            )

        ptr = self.as_mts_tensormap_t()
        self._ptr = None
        return ptr

    def __del__(self):
        if (
            hasattr(self, "_lib")
            and self._lib is not None
            and hasattr(self, "_ptr")
            and hasattr(self, "_parent")
        ):
            if self._parent is None:
                self._lib.mts_tensormap_free(self._ptr)

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, _memodict):
        return self.copy(deep=True)

    def copy(self, deep: bool = True) -> "TensorMap":
        """
        Get a copy of this :py:class:`TensorMap`, with the same keys and blocks. If
        ``deep`` is ``True``, also make a full copy of the blocks values; otherwise, the
        blocks values in the new :py:class:`TensorMap` will share the same memory as
        those in this :py:class:`TensorMap`.

        :param deep: if ``True``, create a deep copy of the blocks in the map
        """
        if deep:
            new_ptr = self._lib.mts_tensormap_copy(self._ptr)
            check_pointer(new_ptr)
            return TensorMap.unsafe_from_ptr(new_ptr)
        else:
            new_blocks = [block.copy(deep=False) for block in self.blocks()]
            return TensorMap(keys=self.keys, blocks=new_blocks)

    def __len__(self):
        return len(self.keys)

    def __repr__(self) -> str:
        return self.print(4)

    def __str__(self) -> str:
        return self.print(-1)

    def __getitem__(self, selection) -> TensorBlock:
        """This is equivalent to self.block(selection)"""
        return self.block(selection)

    # ===== Serialization support ===== #

    @classmethod
    def _from_pickle(cls, buffer: Union[bytes, bytearray]):
        """
        Passed to pickler to reconstruct TensorMap from bytes object
        """
        from .io import create_numpy_array, load_buffer_custom_array

        # TODO: make it so when saving data in torch tensors, we load back data in torch
        # tensors.
        return load_buffer_custom_array(buffer, create_numpy_array)

    def __reduce_ex__(self, protocol: int):
        """
        Used by the Pickler to dump TensorMap object to bytes object. When protocol >= 5
        it supports PickleBuffer which reduces number of copies needed
        """
        from .io import _save_tensor_buffer_raw

        buffer = _save_tensor_buffer_raw(self)
        if protocol >= 5:
            return self._from_pickle, (PickleBuffer(buffer),)
        else:
            return self._from_pickle, (buffer.raw,)

    @staticmethod
    def load(file: Union[str, pathlib.Path, BinaryIO], use_numpy=False) -> "TensorMap":
        """
        Load a serialized :py:class:`TensorMap` from a file or a buffer, calling
        :py:func:`metatensor.load`.

        :param file: file path or file object to load from
        :param use_numpy: should we use the numpy loader or metatensor's. See
            :py:func:`metatensor.load` for more information.
        """
        from .io import load

        return load(file=file, use_numpy=use_numpy)

    @staticmethod
    def load_buffer(
        buffer: Union[bytes, bytearray, memoryview],
        use_numpy=False,
    ) -> "TensorMap":
        """
        Load a serialized :py:class:`TensorMap` from a buffer, calling
        :py:func:`metatensor.io.load_buffer`.

        :param buffer: in-memory buffer containing the data
        :param use_numpy: should we use the numpy loader or metatensor's. See
            :py:func:`metatensor.load` for more information.
        """
        from .io import load_buffer

        return load_buffer(buffer=buffer)

    def save(self, file: Union[str, pathlib.Path, BinaryIO], use_numpy=False):
        """
        Save this :py:class:`TensorMap` to a file or a buffer, calling
        :py:func:`metatensor.save`.

        :param file: file path or file object to save to
        :param use_numpy: should we use the numpy serializer or metatensor's. See
            :py:func:`metatensor.save` for more information.
        """
        from .io import save

        return save(file=file, data=self, use_numpy=use_numpy)

    def save_buffer(self, use_numpy=False) -> memoryview:
        """
        Save this :py:class:`TensorMap` to an in-memory buffer, calling
        :py:func:`metatensor.io.save_buffer`.

        :param use_numpy: should we use numpy serialization or metatensor's. See
            :py:func:`metatensor.save` for more information.
        """
        from .io import save_buffer

        return save_buffer(data=self, use_numpy=use_numpy)

    # ===== Math functions, implemented using metatensor-operations ===== #

    def __eq__(self, other):
        from metatensor.operations import equal

        return equal(self, other)

    def __ne__(self, other):
        from metatensor.operations import equal

        return not equal(self, other)

    def __add__(self, other):
        from metatensor.operations import add

        return add(self, other)

    def __sub__(self, other):
        from metatensor.operations import subtract

        return subtract(self, other)

    def __mul__(self, other):
        from metatensor.operations import multiply

        return multiply(self, other)

    def __matmul__(self, other):
        from metatensor.operations import dot

        return dot(self, other)

    def __truediv__(self, other):
        from metatensor.operations import divide

        return divide(self, other)

    def __pow__(self, other):
        from metatensor.operations import pow

        return pow(self, other)

    def __neg__(self):
        from metatensor.operations import multiply

        return multiply(self, -1)

    def __pos__(self):
        return self

    # ===== Data manipulation ===== #

    @property
    def keys(self) -> Labels:
        """The set of keys labeling the blocks in this tensor map."""
        result = self._lib.mts_tensormap_keys(self._ptr)
        check_pointer(result)
        return Labels.unsafe_from_ptr(result)

    def block_by_id(self, index: int) -> TensorBlock:
        """
        Get the block at ``index`` in this :py:class:`TensorMap`.

        :param index: index of the block to retrieve
        """
        if index >= len(self):
            # we need to raise IndexError to make sure TensorMap supports iterations
            # over blocks with `for block in tensor:` which calls `__getitem__` with
            # integers from 0 to whenever IndexError is raised.
            raise IndexError(
                f"block index out of bounds: we have {len(self)} blocks but the "
                f"index is {index}"
            )

        block = ctypes.POINTER(mts_block_t)()
        self._lib.mts_tensormap_block_by_id(self._ptr, block, index)
        check_pointer(block)
        return TensorBlock.unsafe_view_from_ptr(block, parent=self)

    def blocks_by_id(self, indices: Sequence[int]) -> TensorBlock:
        """
        Get the blocks with the given ``indices`` in this :py:class:`TensorMap`.

        :param indices: indices of the block to retrieve
        """
        return [self.block_by_id(i) for i in indices]

    def block(
        self,
        selection: Union[None, int, Labels, LabelsEntry, Dict[str, int]] = None,
        **kwargs,
    ) -> TensorBlock:
        """
        Get the single block in this :py:class:`TensorMap` matching the ``selection``.

        When ``selection`` is an ``int``, this is equivalent to
        :py:func:`TensorMap.block_by_id`.

        When ``selection`` is a :py:class:`Labels`, :py:class:`LabelsEntry` or
        ``Dict[str, int]``, this function finds the key in this :py:class:`TensorMap`
        with the same values as ``selection`` for the dimensions/names contained in the
        ``selection`` (which can be a subset of the dimensions of the keys); and return
        the corresponding block. This performs a lookup in the keys, so it will be
        slower than :py:func:`TensorMap.block_by_id`, but it is more convenient when the
        position of the block is not known.

        If ``selection`` is :py:obj:`None`, the selection can be passed as keyword
        arguments, which will be converted to a ``Dict[str, int]``.

        :param selection: description of the block to extract

        >>> from metatensor import TensorMap, TensorBlock, Labels
        >>> keys = Labels(["key_1", "key_2"], np.array([[0, 0], [6, 8]]))
        >>> block_1 = TensorBlock(
        ...     values=np.full((3, 5), 1.0),
        ...     samples=Labels.range("sample", 3),
        ...     components=[],
        ...     properties=Labels.range("property", 5),
        ... )
        >>> block_2 = TensorBlock(
        ...     values=np.full((5, 3), 2.0),
        ...     samples=Labels.range("sample", 5),
        ...     components=[],
        ...     properties=Labels.range("property", 3),
        ... )
        >>> tensor = TensorMap(keys, [block_1, block_2])
        >>> # numeric index selection, this gives a block by its position
        >>> block = tensor.block(0)
        >>> block
        TensorBlock
            samples (3): ['sample']
            components (): []
            properties (5): ['property']
            gradients: None
        >>> # This is the first block
        >>> print(block.values.mean())
        1.0
        >>> # use a single key entry (i.e. LabelsEntry) for the selection
        >>> print(tensor.block(tensor.keys[0]).values.mean())
        1.0
        >>> # Labels with a single entry selection
        >>> labels = Labels(names=["key_1", "key_2"], values=np.array([[6, 8]]))
        >>> print(tensor.block(labels).values.mean())
        2.0
        >>> # keyword arguments selection
        >>> print(tensor.block(key_1=0, key_2=0).values.mean())
        1.0
        >>> # dictionary selection
        >>> print(tensor.block({"key_1": 6, "key_2": 8}).values.mean())
        2.0
        """
        if selection is None:
            return self.block(kwargs)
        elif isinstance(selection, int):
            return self.block_by_id(selection)
        else:
            selection = _normalize_selection(selection, like=self.keys)

        keys = self.keys
        matching = keys.select(selection)

        if len(matching) == 0:
            if len(keys) == 0:
                raise ValueError("there are no blocks in this TensorMap")
            else:
                raise ValueError(
                    f"couldn't find any block matching {selection[0].print()}"
                )
        elif len(matching) > 1:
            raise ValueError(
                f"more than one block matched {selection[0].print()}, "
                "use `TensorMap.blocks` to get all of them"
            )
        else:
            return self.block_by_id(matching[0])

    def blocks(
        self,
        selection: Union[
            None, Sequence[int], int, Labels, LabelsEntry, Dict[str, int]
        ] = None,
        **kwargs,
    ) -> List[TensorBlock]:
        """
        Get the blocks in this :py:class:`TensorMap` matching the ``selection``.

        When ``selection`` is ``None`` (the default), all blocks are returned.

        When ``selection`` is an ``int``, this is equivalent to
        :py:func:`TensorMap.block_by_id`; and for a ``List[int]`` this is equivalent to
        :py:func:`TensorMap.blocks_by_id`.

        When ``selection`` is a :py:class:`Labels`, :py:class:`LabelsEntry` or
        ``Dict[str, int]``, this function finds the keys in this :py:class:`TensorMap`
        with the same values as ``selection`` for the dimensions/names contained in the
        ``selection`` (which can be a subset of the dimensions of the keys); and return
        the corresponding blocks. This performs a lookup in the keys, so it will be
        slower than :py:func:`TensorMap.blocks_by_id`, but it is more convenient when
        the position of the blocks is not known.

        If ``selection`` is :py:obj:`None`, the selection can be passed as keyword
        arguments, which will be converted to a ``Dict[str, int]``.

        :param selection: description of the blocks to extract
        """
        if selection is None:
            return self.blocks(kwargs)
        elif isinstance(selection, int):
            return [self.block_by_id(selection)]
        else:
            selection = _normalize_selection(selection, like=self.keys)

        keys = self.keys
        matching = keys.select(selection)

        if len(keys) == 0:
            # return an empty list here instead of the top of this function to make sure
            # the selection was validated
            return []

        if len(matching) == 0:
            raise ValueError(
                f"Couldn't find any block matching '{selection[0].print()}'"
            )
        else:
            return self.blocks_by_id(matching)

    def items(self):
        """get an iterator over (key, block) pairs in this :py:class:`TensorMap`"""
        keys = self.keys
        for i, key in enumerate(keys):
            yield key, self.block_by_id(i)

    def keys_to_samples(
        self,
        keys_to_move: Union[str, Sequence[str]],
        *,
        fill_value=0.0,
        sort_samples=True,
    ) -> "TensorMap":
        """
        Merge blocks along the samples axis, adding ``keys_to_move`` to the end of the
        samples labels dimensions.

        This function will remove ``keys_to_move`` from the keys, and find all blocks
        with the same remaining keys values. It will then merge these blocks along the
        samples direction (i.e. do a *vertical* concatenation), adding ``keys_to_move``
        to the end of the samples labels dimensions. The values taken by
        ``keys_to_move`` in the new samples labels will be the values of these
        dimensions in the merged blocks' keys.

        If ``keys_to_move`` is a set of :py:class:`Labels`, it must be empty
        (``keys_to_move.values.shape[0] == 0``), and only the :py:class:`Labels.names`
        will be used.

        The order of the samples is controlled by ``sort_samples``. If ``sort_samples``
        is true, samples are re-ordered to keep them lexicographically sorted. Otherwise
        they are kept in the order in which they appear in the blocks.

        If the blocks to merge have different property labels, the resulting block
        will have the union of all property labels, and values will be padded with
        the ``fill_value``.

        :param keys_to_move: description of the keys to move
        :param fill_value: scalar value used to fill missing entries in the merged
            blocks. Defaults to 0.0.
        :param sort_samples: whether to sort the merged samples or keep them in the
            order in which they appear in the original blocks
        :return: a new :py:class:`TensorMap` with merged blocks

        .. note::

            The ``fill_value`` also applies to gradient blocks. If using NaN,
            gradient arrays for missing entries will also contain NaN.
        """
        keys_to_move = _normalize_keys_to_move(keys_to_move)
        fill_value_array = _make_fill_value_array(self, fill_value)
        ptr = self._lib.mts_tensormap_keys_to_samples(
            self._ptr, keys_to_move.as_mts_labels_t(), fill_value_array, sort_samples
        )
        check_pointer(ptr)
        return TensorMap.unsafe_from_ptr(ptr)

    def components_to_properties(
        self, dimensions: Union[str, Sequence[str]]
    ) -> "TensorMap":
        """
        Move the given ``dimensions`` from the component labels to the property labels
        for each block.

        :param dimensions: name of the component dimensions to move to the properties
        """
        c_dimensions = _list_or_str_to_array_c_char(dimensions)

        ptr = self._lib.mts_tensormap_components_to_properties(
            self._ptr, c_dimensions, c_dimensions._length_
        )
        check_pointer(ptr)
        return TensorMap.unsafe_from_ptr(ptr)

    def keys_to_properties(
        self,
        keys_to_move: Union[str, Sequence[str], Labels],
        *,
        fill_value=0.0,
        sort_samples=True,
    ) -> "TensorMap":
        """
        Merge blocks along the properties direction, adding ``keys_to_move`` at the
        beginning of the properties labels dimensions.

        This function will remove ``keys_to_move`` from the keys, and find all blocks
        with the same remaining keys values. Then it will merge these blocks along the
        properties direction (i.e. do a *horizontal* concatenation).

        If ``keys_to_move`` is given as strings, then the new property labels will
        **only** contain entries from the existing blocks. For example, merging a block
        with key ``a=0`` and properties ``p=1, 2`` with a block with key ``a=2`` and
        properties ``p=1, 3`` will produce a block with properties ``a, p = (0, 1), (0,
        2), (2, 1), (2, 3)``.

        If ``keys_to_move`` is a set of :py:class:`Labels` and it is empty
        (``len(keys_to_move) == 0``), the :py:class:`Labels.names` will be used as if
        they where passed directly.

        Finally, if ``keys_to_move`` is a non empty set of :py:class:`Labels`, the new
        properties labels will contain **all** of the entries of ``keys_to_move``
        (regardless of the values taken by ``keys_to_move.names`` in the merged blocks'
        keys) followed by the existing properties labels. For example, using ``a=2, 3``
        in ``keys_to_move``, blocks with properties ``p=1, 2`` will result in ``a, p =
        (2, 1), (2, 2), (3, 1), (3, 2)``. If there is no values (no block/missing
        sample) for a given property in the merged block, then the value will be set to
        the ``fill_value``.

        When using a non empty :py:class:`Labels` for ``keys_to_move``, the properties
        labels of all the merged blocks must take the same values.

        The order of the samples in the merged blocks is controlled by ``sort_samples``.
        If ``sort_samples`` is :py:obj:`True`, samples are re-ordered to keep them
        lexicographically sorted. Otherwise they are kept in the order in which they
        appear in the blocks.

        :param keys_to_move: description of the keys to move
        :param fill_value: scalar value used to fill missing entries in the merged
            blocks. Defaults to 0.0.
        :param sort_samples: whether to sort the merged samples or keep them in the
            order in which they appear in the original blocks
        :return: a new :py:class:`TensorMap` with merged blocks

        .. note::

            The ``fill_value`` also applies to gradient blocks. If using NaN,
            gradient arrays for missing entries will also contain NaN.
        """
        keys_to_move = _normalize_keys_to_move(keys_to_move)
        fill_value_array = _make_fill_value_array(self, fill_value)
        ptr = self._lib.mts_tensormap_keys_to_properties(
            self._ptr, keys_to_move.as_mts_labels_t(), fill_value_array, sort_samples
        )
        check_pointer(ptr)
        return TensorMap.unsafe_from_ptr(ptr)

    @property
    def sample_names(self) -> List[str]:
        """
        names of the samples dimensions for all blocks in this :py:class:`TensorMap`
        """
        if len(self.keys) == 0:
            return []

        return self.block(0).samples.names

    @property
    def component_names(self) -> List[str]:
        """
        names of the components dimensions for all blocks in this :py:class:`TensorMap`
        """
        if len(self.keys) == 0:
            return []

        return [c.names[0] for c in self.block(0).components]

    @property
    def property_names(self) -> List[str]:
        """
        names of the properties dimensions for all blocks in this :py:class:`TensorMap`
        """
        if len(self.keys) == 0:
            return []

        return self.block(0).properties.names

    def print(self, max_keys: int) -> str:
        """
        Print this :py:class:`TensorMap` to a string, including at most ``max_keys`` in
        the output.

        :param max_keys: how many keys to include in the output. Use ``-1`` to include
            all keys.
        """

        result = f"TensorMap with {len(self)} blocks\nkeys:"
        result += self.keys.print(max_entries=max_keys, indent=5)
        return result

    @property
    def device(self) -> Device:
        """get the device of all the arrays stored inside this :py:class:`TensorMap`"""
        if len(self.keys) == 0:
            return "cpu"

        return self.block_by_id(0).device

    @property
    def dtype(self) -> DType:
        """get the dtype of all the arrays stored inside this :py:class:`TensorMap`"""
        if len(self.keys) == 0:
            return None

        return self.block_by_id(0).dtype

    @property
    def is_view(self) -> bool:
        """
        Check if this :py:class:`TensorMap` is a view (i.e. does not own the
        underlying data).
        """
        return self._parent is not None

    def to(self, *args, **kwargs) -> "TensorMap":
        """
        Move the keys and all the blocks in this :py:class:`TensorMap` to the given
        ``dtype``, ``device`` and ``arrays`` backend.

        :param dtype: new dtype to use for all arrays. The dtype stays the same if this
            is set to ``None``.
        :param device: new device to use for all arrays. The device stays the same if
            this is set to ``None``.
        :param Optional[str] arrays: new backend to use for the arrays. This can be
            either ``"numpy"``, ``"torch"`` or ``None`` (keeps the existing backend);
            and must be given as a keyword argument (``arrays="numpy"``).
        :param bool non_blocking: If this is ``True`` and the :py:class:`TensorMap`
            contains ``"torch"`` arrays, the function tries to move the data
            asynchronously. See :py:meth:`torch.Tensor.to` for more information.
        """
        arrays = kwargs.pop("arrays", None)
        non_blocking = kwargs.pop("non_blocking", False)
        dtype, device = _data.to_arguments_parse("`TensorMap.to`", *args, **kwargs)

        blocks = []

        for block in self.blocks():
            blocks.append(
                block.to(
                    dtype=dtype,
                    device=device,
                    arrays=arrays,
                    non_blocking=non_blocking,
                )
            )

        tensor = TensorMap(
            self.keys.to(device=device, arrays=arrays, non_blocking=non_blocking),
            blocks,
        )

        for key, value in self.info().items():
            tensor.set_info(key, value)

        return tensor

    def set_info(self, key: str, value: str):
        """
        Set or update the info (i.e. global metadata) ``value`` associated with ``key``
        for this :py:class:`TensorMap`.

        :param key: key of the info
        :param value: value of the info
        """
        self._lib.mts_tensormap_set_info(
            self._ptr, key.encode("utf8"), value.encode("utf8")
        )

    def get_info(self, key: str) -> Optional[str]:
        """
        Get the info (i.e. global metadata) with the given ``key`` for this
        :py:class:`TensorMap`.

        :param key: key of the info to retrieve
        :return: value of the info, or :py:obj:`None` if the info does not exist
        """
        value = ctypes.c_char_p()
        self._lib.mts_tensormap_get_info(self._ptr, key.encode("utf8"), value)

        if value.value:
            return value.value.decode("utf8")
        else:
            return None

    def info(self) -> Dict[str, str]:
        """
        Get all the key/value info pairs stored in this :py:class:`TensorMap`.
        """
        keys = ctypes.POINTER(ctypes.c_char_p)()
        count = c_uintptr_t()
        self._lib.mts_tensormap_info_keys(self._ptr, keys, count)
        result = {}
        for i in range(count.value):
            key = keys[i].decode("utf8")
            result[key] = self.get_info(key)
        return result

    # used by featomic, kept here until we update featomic to use the public API
    @staticmethod
    def _from_ptr(ptr):
        return TensorMap.unsafe_from_ptr(ptr)


def _make_fill_value_array(tensor, fill_value):
    """
    Create an ``mts_array_t`` wrapping a single scalar ``fill_value``.

    The dtype is inferred from the first block's values array. Returns an mts_array_t
    struct suitable for passing to C functions. For empty TensorMaps, returns a
    zero-initialized struct.
    """
    if len(tensor) == 0:
        # Return zero-initialized mts_array_t for empty TensorMaps
        return mts_array_t()

    block = tensor.block_by_id(0)
    values = block.values

    # Always use numpy for the fill_value array. DLPack handles dtype/device
    # conversion when the C API passes it to the array implementation's create
    # callback, so there is no need to match the original array type here.
    fill_value_array = np.array(fill_value, dtype=values.dtype)
    return _data.create_mts_array(fill_value_array)


def _normalize_keys_to_move(keys_to_move: Union[str, Sequence[str], Labels]) -> Labels:
    if isinstance(keys_to_move, str):
        keys_to_move = (keys_to_move,)

    if not isinstance(keys_to_move, Labels):
        for key in keys_to_move:
            assert isinstance(key, str)

        keys_to_move = Labels(
            names=keys_to_move,
            values=np.zeros((0, len(keys_to_move))),
        )

    return keys_to_move


def _list_or_str_to_array_c_char(strings: Union[str, Sequence[str]]):
    if isinstance(strings, str):
        strings = [strings]

    c_strings = ctypes.ARRAY(ctypes.c_char_p, len(strings))()
    for i, v in enumerate(strings):
        assert isinstance(v, str)
        c_strings[i] = v.encode("utf8")

    return c_strings


def _can_cast_to_numpy_int(value):
    return np.can_cast(value, np.int32, casting="same_kind")


def _array_like(values: List[int], like: Labels) -> mts_array_t:
    """
    Convert a list of integers to an array with the same dtype, device and array backend
    as ``like``.

    :param values: list of integers to convert
    :param like: Labels whose array backend and device to match
    :return: a new ``mts_array_t`` with the same dtype, device and array backend as
        ``like``
    """
    np_values = np.array(values, dtype=np.int32).reshape(1, -1)
    cpu_array = _data.create_mts_array(np_values)

    try:
        like_array = like._raw_values

        dl_managed_ptr = ctypes.POINTER(DLManagedTensorVersioned)()
        device = DLDevice(device_type=DLDeviceType.kDLCPU, device_id=0)
        version = DLPackVersion(major=1, minor=0)
        status = cpu_array.as_dlpack(
            cpu_array.ptr,
            ctypes.byref(dl_managed_ptr),
            device,
            None,
            version,
        )
        check_status(status)

        result_array = mts_array_t()
        status = like_array.from_dlpack(
            like_array.ptr,
            dl_managed_ptr,
            ctypes.byref(result_array),
        )
        check_status(status)

        target_device = DLDevice()
        status = like_array.device(like_array.ptr, ctypes.byref(target_device))
        check_status(status)

        if target_device.device_type != DLDeviceType.kDLCPU:
            device_array = mts_array_t()
            status = result_array.copy(
                result_array.ptr,
                target_device,
                ctypes.byref(device_array),
            )
            check_status(status)

            if result_array.destroy:
                result_array.destroy(result_array.ptr)

            result_array = device_array

        return result_array
    finally:
        if cpu_array.destroy:
            cpu_array.destroy(cpu_array.ptr)


def _normalize_selection(
    selection: Union[Labels, LabelsEntry, Dict[str, int]],
    like: Labels,
) -> Labels:
    if isinstance(selection, dict):
        for key, value in selection.items():
            if isinstance(value, int):
                # all good
                pass
            elif isinstance(value, float) or not _can_cast_to_numpy_int(value):
                raise TypeError(
                    f"expected integer values in selection, got {key}={value} of "
                    f"type {type(value)}"
                )

        if len(selection) == 0:
            return Labels([], _array_like([], like))
        else:
            return Labels(
                list(selection.keys()),
                _array_like([np.int32(v) for v in selection.values()], like),
            )

    elif isinstance(selection, Labels):
        return selection

    elif isinstance(selection, LabelsEntry):
        return Labels(selection.names, selection.values.reshape(1, -1))

    else:
        raise TypeError(f"invalid type for block selection: {type(selection)}")
