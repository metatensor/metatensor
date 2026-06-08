import ctypes
import io
import pathlib
from pickle import PickleBuffer
from typing import BinaryIO, List, Optional, Sequence, Tuple, Union, overload

import numpy as np

from . import _data
from ._c_api import (
    c_uintptr_t,
    mts_array_t,
    mts_labels_t,
)
from ._c_lib import _get_library
from ._data import Array, Device
from ._status import check_pointer


try:
    import torch

    torch_int32 = torch.int32
except ImportError:
    torch_int32 = np.int32


class LabelsEntry:
    """A single entry (i.e. row) in a set of :py:class:`Labels`.

    The main way to create a :py:class:`LabelsEntry` is to index a
    :py:class:`Labels` or iterate over them.

    >>> from metatensor import Labels
    >>> import numpy as np
    >>> labels = Labels(
    ...     names=["system", "atom", "center_type"],
    ...     values=np.array([(0, 1, 8), (0, 2, 1), (0, 5, 1)]),
    ... )
    >>> entry = labels[0]  # or labels.entry(0)
    >>> entry.names
    ['system', 'atom', 'center_type']
    >>> print(entry.values)
    [0 1 8]
    """

    def __init__(self, names: List[str], values: Array):
        assert isinstance(names, list)
        for n in names:
            assert isinstance(n, str)
        assert isinstance(values, Array.__supertype__)

        self._names = names

        if len(values.shape) != 1 or values.dtype not in [np.int32, torch_int32]:
            raise ValueError(
                "LabelsEntry values must be a 1-dimensional array of 32-bit integers"
            )

        self._values = values
        self._cached_device = _data.array_device(values)

    @property
    def names(self) -> List[str]:
        """names of the dimensions for this Labels entry"""
        return self._names

    @property
    def values(self) -> Array:
        """
        values associated with each dimension of this Labels entry, stored as
        32-bit integers.
        """
        return self._values

    @property
    def device(self) -> Device:
        """
        Get the device of this labels entry.
        """
        return self._cached_device

    def print(self) -> str:
        """
        print this entry as a named tuple (i.e. ``(key_1=value_1, key_2=value_2)``)
        """
        values = [f"{n}={v}" for n, v in zip(self.names, self.values, strict=True)]
        return f"({', '.join(values)})"

    def __repr__(self) -> str:
        return "LabelsEntry" + self.print()

    def __len__(self) -> int:
        """number of dimensions in this labels entry"""
        return self._values.shape[0]

    def __getitem__(self, dimension: Union[str, int]) -> int:
        """get the value associated with the dimension in this entry"""
        if isinstance(dimension, int):
            return self._values[dimension]
        elif isinstance(dimension, str):
            try:
                i = self._names.index(dimension)
            except ValueError:
                raise ValueError(
                    f"'{dimension}' not found in the dimensions of these Labels"
                )

            return self._values[i]

        else:
            raise TypeError(
                f"can only index LabelsEntry with str or int, got {type(dimension)}"
            )

    def __eq__(self, other: "LabelsEntry") -> bool:
        """
        check if ``self`` and ``other`` are equal (same dimensions/names and
        same values)
        """

        if not isinstance(other, LabelsEntry):
            raise TypeError(
                f"can only compare between LabelsEntry for equality, got {type(other)}"
            )

        return (
            self._names == other._names
            and self._values.shape == other._values.shape
            and np.all(self._values == other._values)
        )

    def __ne__(self, other: "LabelsEntry") -> bool:
        """
        check if ``self`` and ``other`` are not equal (different
        dimensions/names or different values)
        """
        return not self.__eq__(other)


class Labels:
    """
    A set of labels carrying metadata associated with a :py:class:`TensorMap`.

    The metadata can be though as a list of tuples, where each value in the
    tuple also has an associated dimension name. In practice, the dimensions
    ``names`` are stored separately from the ``values``, and the values are in a
    2-dimensional array integers with the shape ``(n_entries, n_dimensions)``.
    Each row/entry in this array is unique, and they are often (but not always)
    sorted in lexicographic order.

    >>> from metatensor import Labels
    >>> import numpy as np
    >>> labels = Labels(
    ...     names=["system", "atom", "center_type"],
    ...     values=np.array([(0, 1, 8), (0, 2, 1), (0, 5, 1)]),
    ... )
    >>> labels
    Labels(
        system  atom  center_type
          0      1         8
          0      2         1
          0      5         1
    )
    >>> labels.names
    ['system', 'atom', 'center_type']
    >>> print(labels.values)
    [[0 1 8]
     [0 2 1]
     [0 5 1]]


    One can also iterate over labels entries, or directly index the :py:class:`Labels`
    to get a specific entry

    >>> entry = labels[0]  # or labels.entry(0)
    >>> entry.names
    ['system', 'atom', 'center_type']
    >>> print(entry.values)
    [0 1 8]
    >>> for entry in labels:
    ...     print(entry)
    LabelsEntry(system=0, atom=1, center_type=8)
    LabelsEntry(system=0, atom=2, center_type=1)
    LabelsEntry(system=0, atom=5, center_type=1)

    Or get all the values associated with a given dimension/column name

    >>> print(labels.column("atom"))
    [1 2 5]
    >>> print(labels["atom"])  # alternative syntax for the above
    [1 2 5]

    Labels can be checked for equality:

    >>> labels == labels
    True


    Finally, it is possible to check if a value is inside (non-view) labels, and
    get the corresponding position:

    >>> labels.position([0, 2, 1])
    1
    >>> print(labels.position([0, 2, 4]))
    None
    >>> (0, 2, 4) in labels
    False
    >>> labels[2] in labels
    True
    """

    def __init__(
        self,
        names: Union[str, Sequence[str]],
        values: np.ndarray,
        assume_unique: bool = False,
    ):
        """
        :param names: names of the dimensions in the new labels. A single string is
            transformed into a list with one element, i.e. ``names="a"`` is the same as
            ``names=["a"]``.
        :param values: values of the labels, this needs to be a 2-dimensional array of
            integers.
        :param assume_unique: skip uniqueness checks inside metatensor. This should only
            be set to ``True`` if you can ensure that label entries are already unique,
            either by construction or because you checked.
        """

        names = _normalize_names_type(names)

        if not isinstance(values, Array.__supertype__):
            raise ValueError("`values` must be a numpy ndarray or torch tensor")

        backend = _data.array_backend(values)

        if len(values) == 0:
            # Explicitly define empty labels
            values = np.empty((0, len(names)), dtype=np.int32)
            values = _data.array_change_backend(values, backend)

        if len(values.shape) != 2:
            # make sure the array is 2D
            raise ValueError("`values` must be a 2D array")

        if len(names) != values.shape[1]:
            raise ValueError(
                "`names` must have an entry for each column of the `values` array"
            )

        if backend == "numpy":
            try:
                # We need to make sure the data has the right type
                values = values.astype(
                    np.int32,
                    casting="same_kind",
                    subok=False,
                    copy=False,
                )
                # values should not be writeable
                values.flags.writeable = False
            except TypeError as e:
                raise TypeError("Labels values must be convertible to integers") from e
        elif backend == "torch":
            if values.requires_grad:
                raise ValueError("Labels values can not require gradients")

            if not _data.array_dtype_is_integer(values):
                raise TypeError("Labels values must be of integer dtype")

            values = values.to(dtype=torch.int32, copy=False)

        if values.ndim == 1:
            values = values.reshape(-1, len(names))

        self._lib = _get_library()
        c_names = ctypes.ARRAY(ctypes.c_char_p, len(names))()
        for i, n in enumerate(names):
            c_names[i] = n.encode("utf8")

        array = _data.create_mts_array(values)

        if assume_unique:
            ptr = self._lib.mts_labels_assume_unique(c_names, len(names), array)
        else:
            ptr = self._lib.mts_labels(c_names, len(names), array)

        check_pointer(ptr)

        self._ptr = ptr
        self._names = names
        self._cached_values = values
        self._cached_device = None

    @staticmethod
    def single() -> "Labels":
        """
        Create :py:class:`Labels` to use when there is no relevant metadata and
        only one entry in the corresponding dimension (e.g. keys when a tensor
        map contains a single block).
        """
        return Labels(names=["_"], values=np.zeros(shape=(1, 1), dtype=np.int32))

    @staticmethod
    def empty(names: Union[str, Sequence[str]]) -> "Labels":
        """
        Create :py:class:`Labels` with given ``names`` but no values.

        :param names: names of the dimensions in the new labels. A single string
                      is transformed into a list with one element, i.e.
                      ``names="a"`` is the same as ``names=["a"]``.
        """
        return Labels(names=names, values=np.empty((0, len(names))))

    @staticmethod
    def range(name: str, end: int) -> "Labels":
        """
        Create :py:class:`Labels` with a single dimension using the given
        ``name`` and values in the ``[0, end)`` range.

        :param name: name of the single dimension in the new labels.
        :param end: end of the range for labels

        >>> from metatensor import Labels
        >>> labels = Labels.range("dummy", 7)
        >>> labels.names
        ['dummy']
        >>> print(labels.values)
        [[0]
         [1]
         [2]
         [3]
         [4]
         [5]
         [6]]
        """
        return Labels(
            names=[name],
            values=np.arange(end, dtype=np.int32).reshape(-1, 1),
        )

    @staticmethod
    def unsafe_from_ptr(labels: ctypes.POINTER(mts_labels_t)):
        """
        Create Labels from a raw ``mts_labels_t`` pointer.

        The :py:class:`Labels` take ownership of the pointer, and will release the
        corresponding memory when garbage-collected.
        """
        assert labels, "mts_labels_t pointer is null"

        obj = Labels.__new__(Labels)
        obj._lib = _get_library()
        obj._ptr = labels

        names_ptr = ctypes.POINTER(ctypes.c_char_p)()
        names_count = c_uintptr_t()
        obj._lib.mts_labels_dimensions(labels, names_ptr, names_count)

        names = []
        for i in range(names_count.value):
            names.append(names_ptr[i].decode("utf8"))
        obj._names = names

        obj._cached_values = None
        obj._cached_device = None

        return obj

    def release(self) -> ctypes.POINTER(mts_labels_t):
        """
        Release the underlying C pointer of these :py:class:`Labels`.

        This class is no longer managing the labels memory after the call, the user is
        expected to re-create Labels with :py:meth:`Labels.unsafe_from_ptr`, or pass the
        pointer to a C function that will call `mts_labels_free`.
        """
        ptr = self.as_mts_labels_t()
        self._ptr = None
        return ptr

    def as_mts_labels_t(self) -> ctypes.POINTER(mts_labels_t):
        """
        Get the underlying C pointer for these :py:class:`Labels`. This class still
        manages the labels memory after the call. Use :py:meth:`Labels.release` to
        take ownership of the pointerr
        """
        if not self._ptr:
            raise RuntimeError("can not access these Labels, they have been released")

        return self._ptr

    def __del__(self):
        if hasattr(self, "_lib") and self._lib is not None:
            if hasattr(self, "_labels") and self._ptr is not None:
                self._lib.mts_labels_free(self._ptr)

    def __deepcopy__(self, _memodict):
        ptr = self._lib.mts_labels_clone(self.as_mts_labels_t())
        check_pointer(ptr)
        return Labels.unsafe_from_ptr(ptr)

    def __copy__(self):
        return self.__deepcopy__({})

    def __str__(self) -> str:
        if not self._ptr:
            # The Labels have been released
            return "Labels(<empty>)"

        printed = self.print(4, 3)
        return f"Labels(\n   {printed}\n)"

    def __repr__(self) -> str:
        if not self._ptr:
            # The Labels have been released
            return "Labels(<empty>)"

        printed = self.print(-1, 3)
        return f"Labels(\n   {printed}\n)"

    def __len__(self) -> int:
        """number of entries in these labels"""
        return self.values.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield LabelsEntry(self._names, self.values[i, :])

    @overload
    def __getitem__(self, dimension: str) -> np.ndarray:
        pass

    @overload
    def __getitem__(self, index: int) -> LabelsEntry:
        pass

    def __getitem__(self, index):
        """
        When indexing with a string, get the values for the corresponding dimension as a
        1-dimensional array (i.e. :py:func:`Labels.column`).

        When indexing with an integer, get the corresponding row/labels entry (i.e.
        :py:func:`Labels.entry`).
        """
        if isinstance(index, (int, np.int8, np.int16, np.int32, np.int64)):
            return self.entry(index)
        if isinstance(index, (str, int)):
            return self.column(index) if isinstance(index, str) else self.entry(index)

        raise TypeError(
            "Labels can only be indexed by a single string or integer"
            f", got {type(index)}."
            "To select multiple columns, construct a new `Labels` object."
        )

    def __contains__(
        self,
        entry: Union[LabelsEntry, Sequence[int]],
    ) -> bool:
        """check if these :py:class:`Labels` contain the given ``entry``"""
        return self.position(entry) is not None

    def __eq__(self, other: "Labels") -> bool:
        """
        check if two set of labels are equal (same dimension names and same
        values)
        """
        if not isinstance(other, Labels):
            raise TypeError(
                f"can only compare between Labels for equality, got {type(other)}"
            )

        if self._names != other._names or self.values.shape != other.values.shape:
            return False

        self_is_torch = _data.array_backend(self.values) == "torch"
        other_is_torch = _data.array_backend(other.values) == "torch"

        if self_is_torch and other_is_torch:
            return bool(torch.all(self.values == other.values))
        elif self_is_torch:
            return bool(np.all(self.values.cpu().numpy() == other.values))
        elif other_is_torch:
            return bool(np.all(self.values == other.values.cpu().numpy()))
        else:
            return bool(np.all(self.values == other.values))

    def __ne__(self, other: "Labels") -> bool:
        """
        check if two set of labels are not equal (different dimension names or
        different values)
        """
        return not self.__eq__(other)

    # ===== Serialization support ===== #

    @classmethod
    def _from_pickle(cls, buffer: Union[bytes, bytearray]):
        """
        Passed to pickler to reconstruct Labels from bytes object
        """
        from .io import load_labels_buffer

        return load_labels_buffer(buffer)

    def __reduce_ex__(self, protocol: int):
        """
        Used by the Pickler to dump Labels object to bytes object. When protocol >= 5
        it supports PickleBuffer which reduces number of copies needed
        """
        from .io import _save_labels_buffer_raw

        buffer = _save_labels_buffer_raw(self)
        if protocol >= 5:
            return self._from_pickle, (PickleBuffer(buffer),)
        else:
            return self._from_pickle, (buffer.raw,)

    @staticmethod
    def load(file: Union[str, pathlib.Path, BinaryIO]) -> "Labels":
        """
        Load a serialized :py:class:`Labels` from a file, calling
        :py:func:`metatensor.load_labels`.

        :param file: file path or file object to load from
        """
        from .io import load_labels

        return load_labels(file=file)

    @staticmethod
    def load_buffer(buffer: Union[bytes, bytearray, memoryview]) -> "Labels":
        """
        Load a serialized :py:class:`Labels` from a buffer, calling
        :py:func:`metatensor.io.load_labels_buffer`.

        :param buffer: in-memory buffer containing the data
        """
        from .io import load_labels_buffer

        return load_labels_buffer(buffer=buffer)

    def save(self, file: Union[str, pathlib.Path, BinaryIO]):
        """
        Save these :py:class:`Labels` to a file, calling :py:func:`metatensor.save`.

        :param file: file path or file object to save to
        """
        from .io import save

        return save(file=file, data=self)

    def save_buffer(self) -> memoryview:
        """
        Save these :py:class:`Labels` to an in-memory buffer, calling
        :py:func:`metatensor.io.save_buffer`.
        """
        from .io import save_buffer

        return save_buffer(data=self)

    # ===== Data manipulation ===== #

    @property
    def names(self) -> List[str]:
        """names of the dimensions for these :py:class:`Labels`"""
        return self._names

    @property
    def _raw_values(self) -> mts_array_t:
        """Get the raw ``mts_array_t`` corresponding to these Labels' values"""
        data = mts_array_t()
        self._lib.mts_labels_values(self.as_mts_labels_t(), data)
        return data

    @property
    def values(self) -> Array:
        """
        values associated with each dimensions of the :py:class:`Labels`, stored
        as 2-dimensional tensor of 32-bit integers
        """
        if self._cached_values is None:
            self._cached_values = _data.mts_array_to_python_array(
                self._raw_values, self
            )

        return self._cached_values

    def append(self, name: str, values: np.ndarray) -> "Labels":
        """Append a new dimension to the end of the :py:class:`Labels`.

        :param name: name of the new dimension
        :param values: 1D array of values for the new dimension

        >>> import numpy as np
        >>> from metatensor import Labels
        >>> label = Labels("foo", np.array([[42]]))
        >>> label
        Labels(
            foo
            42
        )
        >>> label.append(name="bar", values=np.array([10]))
        Labels(
            foo  bar
            42   10
        )
        """
        return self.insert(index=len(self), name=name, values=values)

    def insert(self, index: int, name: str, values: np.ndarray) -> "Labels":
        """Insert a new dimension before ``index`` in the :py:class:`Labels`.

        :param index: index before the new dimension is inserted
        :param name: name of the new dimension
        :param values: 1D array of values for the new dimension

        >>> import numpy as np
        >>> from metatensor import Labels
        >>> label = Labels("foo", np.array([[42]]))
        >>> label
        Labels(
            foo
            42
        )
        >>> label.insert(0, name="bar", values=np.array([10]))
        Labels(
            bar  foo
            10   42
        )
        """
        new_names = self.names.copy()
        new_names.insert(index, name)

        if not isinstance(values, np.ndarray):
            raise ValueError("`values` must be a numpy ndarray")

        if len(values.shape) != 1:
            raise ValueError("`values` must be a 1D array")

        if values.shape[0] != len(self):
            raise ValueError(
                f"the new `values` contains {values.shape[0]} entries, "
                f"but the Labels contains {len(self)}"
            )

        new_values = np.insert(self.values, index, values, axis=1)

        return Labels(names=new_names, values=new_values)

    def permute(self, dimensions_indexes: List[int]) -> "Labels":
        """Permute dimensions according to ``dimensions_indexes`` in the
        :py:class:`Labels`.

        :param dimensions_indexes: desired ordering of the dimensions
        :raises ValueError: if length of ``dimensions_indexes`` does not match the
            Labels length
        :raises ValueError: if duplicate values are present in ``dimensions_indexes``

        >>> import numpy as np
        >>> from metatensor import Labels
        >>> label = Labels(["foo", "bar", "baz"], np.array([[42, 10, 3]]))
        >>> label
        Labels(
            foo  bar  baz
            42   10    3
        )
        >>> label.permute([2, 0, 1])
        Labels(
            baz  foo  bar
             3   42   10
        )
        """
        if len(dimensions_indexes) != len(self.names):
            raise ValueError(
                f"the length of `dimensions_indexes` ({len(dimensions_indexes)}) does "
                f"not match the number of dimensions in the Labels ({len(self.names)})"
            )

        names = [self.names[d] for d in dimensions_indexes]

        return Labels(names=names, values=self.values[:, dimensions_indexes])

    def remove(self, name: str) -> "Labels":
        """Remove ``name`` from the dimensions of the :py:class:`Labels`.

        Removal can only be performed if the resulting :py:class:`Labels` instance will
        be unique.

        :param name: name to be removed
        :raises ValueError: if the name is not present.

        >>> import numpy as np
        >>> from metatensor import Labels
        >>> label = Labels(["foo", "bar"], np.array([[42, 10]]))
        >>> label
        Labels(
            foo  bar
            42   10
        )
        >>> label.remove(name="bar")
        Labels(
            foo
            42
        )

        If the new :py:class:`Labels` is not unique an error is raised.

        >>> from metatensor import MetatensorError
        >>> label = Labels(["foo", "bar"], np.array([[42, 10], [42, 11]]))
        >>> label
        Labels(
            foo  bar
            42   10
            42   11
        )
        >>> try:
        ...     label.remove(name="bar")
        ... except MetatensorError as e:
        ...     print(e)
        invalid parameter: can not have the same label entry multiple times: [42] is already present
        """  # noqa E501
        if name not in self.names:
            raise ValueError(f"'{name}' not found in the dimensions of these Labels")

        new_names = self.names.copy()
        new_names.remove(name)

        index = self.names.index(name)
        new_values = np.delete(self.values, index, axis=1)

        return Labels(names=new_names, values=new_values)

    def rename(self, old: str, new: str) -> "Labels":
        """Rename the ``old`` dimension to ``new`` in the :py:class:`Labels`.

        :param old: name to be replaced
        :param new: name after the replacement
        :raises ValueError: if old is not present.

        >>> import numpy as np
        >>> from metatensor import Labels
        >>> label = Labels("foo", np.array([[42]]))
        >>> label
        Labels(
            foo
            42
        )
        >>> label.rename("foo", "bar")
        Labels(
            bar
            42
        )
        """

        if old not in self.names:
            raise ValueError(f"'{old}' not found in the dimensions of these Labels")

        names = self.names.copy()
        index = names.index(old)
        names[index] = new

        return Labels(names=names, values=self.values)

    def to(self, *args, **kwargs) -> "Labels":
        """
        Move the values of these Labels to the given ``dtype``, ``device`` and
        ``arrays`` backend.

        :param dtype: new dtype to use for all arrays. The dtype stays the same if this
            is set to ``None``.
        :param device: new device to use for all arrays. The device stays the same if
            this is set to ``None``.
        :param Optional[str] arrays: new backend to use for the arrays. This can be
            either ``"numpy"``, ``"torch"`` or ``None`` (keeps the existing backend);
            and must be given as a keyword argument (``arrays="numpy"``).
        :param bool non_blocking: If this is ``True`` and the :py:class:`TensorBlock`
            contains ``"torch"`` arrays, the function tries to move the data
            asynchronously. See :py:meth:`torch.Tensor.to` for more information.
        """
        arrays = kwargs.pop("arrays", None)
        non_blocking = kwargs.pop("non_blocking", False)
        dtype, device = _data.to_arguments_parse("`Labels.to`", *args, **kwargs)

        if dtype is not None:
            raise ValueError("Labels values must be int32, `dtype` can not be changed")

        already_there = True
        if arrays is not None:
            already_there = already_there and arrays == _data.array_backend(self.values)

        if device is not None:
            already_there = already_there and str(device) == str(self.device)

        if already_there:
            # nothing to do
            return self

        # move the data as required
        values = self.values

        if arrays is not None:
            values = _data.array_change_backend(values, arrays)

        if device is not None:
            values = _data.array_change_device(
                values, device, non_blocking=non_blocking
            )

        return Labels(names=self.names, values=values, assume_unique=True)

    @property
    def device(self) -> Device:
        """
        Get the device of these Labels.
        """
        if self._cached_device is None:
            self._cached_device = _data.array_device(self.values)

        return self._cached_device

    def position(self, entry: Union[LabelsEntry, Sequence[int]]) -> Optional[int]:
        """
        Get the position of the given ``entry`` in this set of
        :py:class:`Labels`, or ``None`` if the entry is not present in the
        labels.
        """

        result = ctypes.c_int64()
        c_entry = ctypes.ARRAY(ctypes.c_int32, len(entry))()
        for i, v in enumerate(entry):
            c_entry[i] = ctypes.c_int32(v)

        self._lib.mts_labels_position(
            self.as_mts_labels_t(),
            c_entry,
            c_entry._length_,
            result,
        )

        if result.value >= 0:
            return result.value
        else:
            return None

    def difference(self, other: "Labels") -> "Labels":
        """
        Take the set difference of these :py:class:`Labels` with ``other``.

        If you want to know where entries in ``self`` and ``other`` ends up in the
        difference, you can use :py:meth:`Labels.difference_and_mapping`.

        >>> import numpy as np
        >>> from metatensor import Labels
        >>> first = Labels(
        ...     names=["a", "b"], values=np.array([[0, 1], [1, 3], [0, 3], [2, 2]])
        ... )
        >>> second = Labels(
        ...     names=["a", "b"], values=np.array([[0, 3], [1, 3], [1, 2], [2, 1]])
        ... )
        >>> first.difference(second)
        Labels(
            a  b
            0  1
            2  2
        )
        """
        output = ctypes.POINTER(mts_labels_t)()
        self._lib.mts_labels_difference(
            self.as_mts_labels_t(),
            other.as_mts_labels_t(),
            ctypes.pointer(output),
            None,
            0,
        )

        check_pointer(output)
        return Labels.unsafe_from_ptr(output)

    def difference_and_mapping(self, other: "Labels") -> Tuple["Labels", np.ndarray]:
        """
        Take the set difference of these :py:class:`Labels` with ``other``.

        This function also returns the position in the difference where each entry of
        the input :py:class::`Labels` ended up.

        :return: Tuple containing the difference, and a :py:class:`numpy.ndarray`
            containing the position in the difference of the entries from ``self``.

        >>> import numpy as np
        >>> from metatensor import Labels
        >>> first = Labels(
        ...     names=["a", "b"], values=np.array([[0, 1], [1, 3], [0, 3], [2, 2]])
        ... )
        >>> second = Labels(
        ...     names=["a", "b"], values=np.array([[0, 3], [1, 3], [1, 2], [2, 1]])
        ... )
        >>> difference, mapping_1 = first.difference_and_mapping(second)
        >>> difference
        Labels(
            a  b
            0  1
            2  2
        )
        >>> print(mapping_1)
        [ 0 -1 -1  1]
        """
        output = ctypes.POINTER(mts_labels_t)()
        first_mapping = np.zeros(len(self), dtype=np.int64)

        self._lib.mts_labels_difference(
            self.as_mts_labels_t(),
            other.as_mts_labels_t(),
            ctypes.pointer(output),
            first_mapping.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            len(first_mapping),
        )

        check_pointer(output)
        return Labels.unsafe_from_ptr(output), first_mapping

    def union(self, other: "Labels") -> "Labels":
        """
        Take the union of these :py:class:`Labels` with ``other``.

        If you want to know where entries in ``self`` and ``other`` ends up in the
        union, you can use :py:meth:`Labels.union_and_mapping`.

        >>> import numpy as np
        >>> from metatensor import Labels
        >>> first = Labels(names=["a", "b"], values=np.array([[0, 1], [1, 2], [0, 3]]))
        >>> second = Labels(names=["a", "b"], values=np.array([[0, 3], [1, 3], [1, 2]]))
        >>> first.union(second)
        Labels(
            a  b
            0  1
            1  2
            0  3
            1  3
        )
        """
        output = ctypes.POINTER(mts_labels_t)()
        self._lib.mts_labels_union(
            self.as_mts_labels_t(),
            other.as_mts_labels_t(),
            ctypes.pointer(output),
            None,
            0,
            None,
            0,
        )

        check_pointer(output)
        return Labels.unsafe_from_ptr(output)

    def union_and_mapping(
        self, other: "Labels"
    ) -> Tuple["Labels", np.ndarray, np.ndarray]:
        """
        Take the union of these :py:class:`Labels` with ``other``.

        This function also returns the position in the union where each entry of the
        input :py:class::`Labels` ended up.

        :return: Tuple containing the union, a :py:class:`numpy.ndarray` containing the
            position in the union of the entries from ``self``, and a
            :py:class:`numpy.ndarray` containing the position in the union of the
            entries from ``other``.

        >>> import numpy as np
        >>> from metatensor import Labels
        >>> first = Labels(names=["a", "b"], values=np.array([[0, 1], [1, 2], [0, 3]]))
        >>> second = Labels(names=["a", "b"], values=np.array([[0, 3], [1, 3], [1, 2]]))
        >>> union, mapping_1, mapping_2 = first.union_and_mapping(second)
        >>> union
        Labels(
            a  b
            0  1
            1  2
            0  3
            1  3
        )
        >>> print(mapping_1)
        [0 1 2]
        >>> print(mapping_2)
        [2 3 1]
        """
        output = ctypes.POINTER(mts_labels_t)()
        first_mapping = np.zeros(len(self), dtype=np.int64)
        second_mapping = np.zeros(len(other), dtype=np.int64)

        self._lib.mts_labels_union(
            self.as_mts_labels_t(),
            other.as_mts_labels_t(),
            ctypes.pointer(output),
            first_mapping.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            len(first_mapping),
            second_mapping.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            len(second_mapping),
        )

        check_pointer(output)
        return Labels.unsafe_from_ptr(output), first_mapping, second_mapping

    def intersection(self, other: "Labels") -> "Labels":
        """
        Take the intersection of these :py:class:`Labels` with ``other``.

        If you want to know where entries in ``self`` and ``other`` ends up in the
        intersection, you can use :py:meth:`Labels.intersection_and_mapping`.

        >>> import numpy as np
        >>> from metatensor import Labels
        >>> first = Labels(names=["a", "b"], values=np.array([[0, 1], [1, 2], [0, 3]]))
        >>> second = Labels(names=["a", "b"], values=np.array([[0, 3], [1, 3], [1, 2]]))
        >>> first.intersection(second)
        Labels(
            a  b
            1  2
            0  3
        )
        """
        output = ctypes.POINTER(mts_labels_t)()
        self._lib.mts_labels_intersection(
            self.as_mts_labels_t(),
            other.as_mts_labels_t(),
            ctypes.pointer(output),
            None,
            0,
            None,
            0,
        )

        check_pointer(output)
        return Labels.unsafe_from_ptr(output)

    def intersection_and_mapping(
        self, other: "Labels"
    ) -> Tuple["Labels", np.ndarray, np.ndarray]:
        """
        Take the intersection of these :py:class:`Labels` with ``other``.

        This function also returns the position in the intersection where each entry of
        the input :py:class:`Labels` ended up.

        :return: Tuple containing the intersection, a :py:class:`numpy.ndarray`
            containing the position in the intersection of the entries from ``self``,
            and a :py:class:`numpy.ndarray` containing the position in the intersection
            of the entries from ``other``. If entries in ``self`` or ``other`` are not
            used in the output, the mapping for them is set to ``-1``.

        >>> import numpy as np
        >>> from metatensor import Labels
        >>> first = Labels(names=["a", "b"], values=np.array([[0, 1], [1, 2], [0, 3]]))
        >>> second = Labels(names=["a", "b"], values=np.array([[0, 3], [1, 3], [1, 2]]))
        >>> intersection, mapping_1, mapping_2 = first.intersection_and_mapping(second)
        >>> intersection
        Labels(
            a  b
            1  2
            0  3
        )
        >>> print(mapping_1)
        [-1  0  1]
        >>> print(mapping_2)
        [ 1 -1  0]
        """
        output = ctypes.POINTER(mts_labels_t)()
        first_mapping = np.zeros(len(self), dtype=np.int64)
        second_mapping = np.zeros(len(other), dtype=np.int64)

        self._lib.mts_labels_intersection(
            self.as_mts_labels_t(),
            other.as_mts_labels_t(),
            ctypes.pointer(output),
            first_mapping.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            len(first_mapping),
            second_mapping.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            len(second_mapping),
        )

        check_pointer(output)
        return Labels.unsafe_from_ptr(output), first_mapping, second_mapping

    def select(self, selection: "Labels") -> np.ndarray:
        """
        Select entries in these :py:class:`Labels` that match the ``selection``.

        The selection's names must be a subset of the names of these labels.

        All entries in these :py:class:`Labels` that match one of the entry in the
        ``selection`` for all the selection's dimension will be picked. Any entry in the
        ``selection`` but not in these :py:class:`Labels` will be ignored.

        >>> import numpy as np
        >>> from metatensor import Labels
        >>> labels = Labels(
        ...     names=["a", "b"],
        ...     values=np.array([[0, 1], [1, 2], [0, 3], [1, 1], [2, 4]]),
        ... )
        >>> selection = Labels(names=["a"], values=np.array([[0], [2], [5]]))
        >>> print(labels.select(selection))
        [0 2 4]

        :param selection: description of the entries to select
        :return: 1-dimensional ndarray containing the integer indices of selected
            entries
        """
        selected = np.zeros((len(self)), dtype=ctypes.c_uint64)
        selected_count = c_uintptr_t(len(self))

        self._lib.mts_labels_select(
            self.as_mts_labels_t(),
            selection.as_mts_labels_t(),
            selected.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.pointer(selected_count),
        )

        selected.resize(selected_count.value, refcheck=False)

        return selected

    def print(self, max_entries: int, indent: int = 0) -> str:
        """print these :py:class:`Labels` to a string

        :param max_entries: how many entries to print, use ``-1`` to print everything
        :param indent: indent the output by ``indent`` spaces
        """
        return _print_labels(
            self._names,
            self.values,
            max_entries=max_entries,
            indent=indent,
        )

    def entry(self, index: int) -> LabelsEntry:
        """
        Get a single entry/row in these labels.

        .. seealso::

            :py:func:`Labels.__getitem__` as the main way to use this function
        """
        return LabelsEntry(self._names, self.values[index, :])

    def column(self, dimension: str) -> np.ndarray:
        """
        Get the values associated with a single dimension in these labels (i.e. a single
        column of :py:attr:`Labels.values`) as a 1-dimensional array.

        .. seealso::

            :py:func:`Labels.__getitem__` as the main way to use this function
        """
        if not isinstance(dimension, str):
            message = (
                "Labels can only be indexed by a single string or integer, "
                f"got {type(dimension)}"
            )
            raise TypeError(message)

        try:
            index = self.names.index(dimension)
        except ValueError:
            raise ValueError(
                f"'{dimension}' not found in the dimensions of these Labels"
            )

        return self.values[:, index]


def _normalize_names_type(names: Union[str, Sequence[str]]) -> List[str]:
    """
    Transform Labels names from any of the accepted types into the canonical
    representation (list of strings).
    """

    if isinstance(names, str):
        if len(names) == 0:
            names = []
        else:
            names = [names]
    else:
        try:
            names = list(names)
        except TypeError:
            raise TypeError(
                f"Labels names must be a sequence of strings, got {type(names)} instead"
            )

        for name in names:
            if not isinstance(name, str):
                raise TypeError(
                    f"Labels names must be strings, got {type(name)} instead"
                )

    return names


def _print_string_center(output, string, width, last):
    delta = width - len(string)
    n_before = delta // 2
    n_after = delta - n_before

    if last:
        # don't add spaces after the last element
        output.write(" " * n_before)
        output.write(string)
    else:
        output.write(" " * n_before)
        output.write(string)
        output.write(" " * n_after)


def _print_labels(
    names: List[str],
    values: np.ndarray,
    max_entries: int,
    indent: int,
) -> str:
    # ================================================================================ #
    # Step 1: determine the width of all the columns (at least the width of the names  #
    # plus 2, might be wider for large values)                                         #
    # ================================================================================ #

    # the +2 is here use at least one space on each side of the name
    widths = [len(n) + 2 for n in names]

    # first set of values to print (before the "...")
    values_first = []
    # second set of values to print (after the "...")
    values_second = []

    n_elements = values.shape[0]

    # first, determine the width of each column by looking through all the
    # values and names lengths
    if max_entries < 0 or n_elements <= max_entries:
        for entry in values:
            entry_strings = []
            for i, e in enumerate(entry):
                element_str = str(e)
                entry_strings.append(element_str)
                widths[i] = max(widths[i], len(element_str) + 2)

            values_first.append(entry_strings)
    else:
        if max_entries < 2:
            max_entries = 2

        n_after = max_entries // 2
        n_before = max_entries - n_after

        # values before the "..."
        for entry in values[:n_before, :]:
            entry_strings = []
            for i, e in enumerate(entry):
                element_str = str(e)
                entry_strings.append(element_str)
                widths[i] = max(widths[i], len(element_str) + 2)

            values_first.append(entry_strings)

        # values after the "..."
        for entry in values[n_elements - n_after :, :]:
            entry_strings = []
            for i, e in enumerate(entry):
                element_str = str(e)
                entry_strings.append(element_str)
                widths[i] = max(widths[i], len(element_str) + 2)

            values_second.append(entry_strings)

    # ================================================================================ #
    # Step 2: actually create the output string, using io.StringIO to incrementally    #
    # append to the output                                                             #
    # ================================================================================ #

    indent_str = " " * indent
    n_dimensions = values.shape[1]

    output = io.StringIO()
    for i in range(n_dimensions):
        last = i == n_dimensions - 1
        _print_string_center(output, names[i], widths[i], last)
    output.write("\n")

    for strings in values_first:
        output.write(indent_str)
        for i in range(n_dimensions):
            last = i == n_dimensions - 1
            _print_string_center(output, strings[i], widths[i], last)
        output.write("\n")

    if len(values_second) != 0:
        half_header_widths = sum(widths) // 2
        if half_header_widths > 3:
            # 3 characters in '...'
            half_header_widths -= 3

        output.write(indent_str)
        output.write((half_header_widths + 1) * " ")
        output.write("...\n")

        for strings in values_second:
            output.write(indent_str)
            for i in range(n_dimensions):
                last = i == n_dimensions - 1
                _print_string_center(output, strings[i], widths[i], last)
            output.write("\n")

    output = output.getvalue()
    assert output[-1] == "\n"
    return output[:-1]
