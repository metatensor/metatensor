import ctypes
from collections import namedtuple
from typing import List, Optional, Union

import numpy as np

from ._c_api import eqs_labels_t
from ._c_lib import _get_library
from .utils import _ptr_to_const_ndarray


class Labels(np.ndarray):
    """
    A set of labels used to carry metadata associated with a tensor map.

    This is similar to a list of ``n_entries`` named tuples, but stored as a 2D
    array of shape ``(n_entries, n_dimensions)``, with a set of names associated
    with the columns of this array (often called *dimensions*). Each row/entry in
    this array is unique, and they are often (but not always) sorted in
    lexicographic order.

    In Python, :py:class:`Labels` are implemented as a wrapper around a 2D
    ``numpy.ndarray`` with a custom ``dtype`` allowing direct access to the
    different columns:

    .. code-block:: python

        labels = Labels(
            names=["structure", "atom", "center_species"],
            values=...,
        )

        # access all values in a column by name
        structures = labels["structures"]

        # multiple columns at once
        data = labels[["structures", "center_species"]]

        # we can still use all the usual numpy operations
        unique_structures = np.unique(labels["structures"])

    One can also check for the presence of a given entry in Labels, but only if
    the Labels come from a :py:class:`TensorBlock` or a :py:class:`TensorMap`.

    .. code-block:: python

        # create a block in some way

        samples = block.samples

        position = samples.position((1, 3))
        # either None if the sample is not there or the
        # position in the samples of (1, 3)

        # this also works with __contains__
        if (1, 3) in samples:
            ...
    """

    def __new__(cls, names: Union[List[str], str], values: np.ndarray, **kwargs):
        """
        :param names: names of the dimensions in the new labels, in the case of a single
                      name also a single string can be given: ``names = "name"``
        :param values: values of the dimensions, this needs to be a 2D array of
            ``np.int32`` values
        """

        for key in kwargs.keys():
            if key != "_eqs_labels_t":
                raise ValueError(f"unexpected kwarg to Labels: {key}")

        if isinstance(names, str):
            if len(names) == 0:
                names = tuple()
            else:
                names = tuple(names)
        else:
            names = tuple(str(n) for n in names)

        if not isinstance(values, np.ndarray):
            raise ValueError("values parameter must be a numpy ndarray")

        if len(values) == 0:
            # if empty array of values we use the required correct 2d shape
            values = np.zeros((0, len(names)), dtype=np.int32)
        elif len(values.shape) != 2:
            raise ValueError("values parameter must be a 2D array")

        if len(names) != values.shape[1]:
            raise ValueError(
                "names parameter must have an entry for each column of the array"
            )

        try:
            values = np.ascontiguousarray(
                values.astype(
                    np.int32,
                    order="C",
                    casting="same_kind",
                    subok=False,
                    copy=False,
                )
            )
        except TypeError as e:
            raise TypeError("Labels values must be convertible to integers") from e

        dtype = [(name, np.int32) for name in names]

        if values.shape[1] != 0:
            values = values.view(dtype=dtype).reshape((values.shape[0],))

        obj = values.view(cls)

        obj._lib = _get_library()

        obj._eqs_labels_t = kwargs.get("_eqs_labels_t")
        if obj._eqs_labels_t is not None:
            # ensure we have a valid Rust pointer
            assert obj._eqs_labels_t.internal_ptr_ is not None
        else:
            # create a new Rust pointer for these Labels
            obj._eqs_labels_t = _eqs_labels_view(obj)
            obj._lib.eqs_labels_create(obj._eqs_labels_t)

        return obj

    def __array_finalize__(self, obj):
        # do not keep the Rust pointer around around, since one could be taking only a
        # subset of the dimensions (`samples[["structure", "center"]]`) and this
        # would break `position` and `__contains__`
        self._eqs_labels_t = None

        self._lib = getattr(obj, "_lib", None)

    def __del__(self):
        if (
            hasattr(self, "_lib")
            and self._lib is not None
            and hasattr(self, "_eqs_labels_t")
        ):
            self._lib.eqs_labels_free(self._eqs_labels_t)

    @property
    def names(self) -> List[str]:
        """Names of the columns/dimensions used for these labels"""
        return self.dtype.names or []

    @staticmethod
    def single() -> "Labels":
        """
        Get the labels to use when there is no relevant metadata and only one
        entry in the corresponding dimension (e.g. keys when a tensor map
        contains a single block).
        """
        return Labels(names=["_"], values=np.zeros(shape=(1, 1), dtype=np.int32))

    @staticmethod
    def empty(names) -> "Labels":
        """Label with given names but no values.

        :param names: names of the dimensions in the new labels, in the case of a single
                      name also a single string can be given: ``names = "name"``
        """
        return Labels(names=names, values=np.array([]))

    def as_namedtuples(self):
        """
        Iterator over the entries in these Labels as namedtuple instances.

        .. code-block:: python

            labels = Labels(
                names=["structure", "atom", "center_species"],
                values=np.array([[0, 2, 4]], dtype=np.int32),
            )

            for label in labels.as_namedtuples():
                print(label)
                print(label.as_dict())

            # outputs
            # LabelTuple(structure=0, atom=2, center_species=4)
            # {'structure': 0, 'atom': 2, 'center_species': 4}
        """
        named_tuple_class = namedtuple("LabelTuple", self.names)
        named_tuple_class.as_dict = named_tuple_class._asdict

        for entry in self:
            yield named_tuple_class(*entry)

    def _as_eqs_labels_t(self):
        """transform these labels into eqs_labels_t"""
        if self._eqs_labels_t is None:
            return _eqs_labels_view(self)
        else:
            return self._eqs_labels_t

    @staticmethod
    def _from_eqs_labels_t(eqs_labels):
        """
        Convert an eqs_labels_t into a Labels instance.
        """
        names = []
        for i in range(eqs_labels.size):
            names.append(eqs_labels.names[i].decode("utf8"))

        if eqs_labels.count != 0:
            shape = (eqs_labels.count, eqs_labels.size)
            values = _ptr_to_const_ndarray(
                ptr=eqs_labels.values, shape=shape, dtype=np.int32
            )
            values.flags.writeable = False
            return Labels(names, values, _eqs_labels_t=eqs_labels)
        else:
            return Labels(
                names=names,
                values=np.empty(shape=(0, len(names)), dtype=np.int32),
            )

    def position(self, label) -> Optional[int]:
        """
        Get the position of the given ``label`` entry in this set of labels.

        This is only available if the labels comes from a
        :py:class:`TensorBlock` or a :py:class:`TensorMap`. If you need it for
        standalone labels, please let us know!
        """
        lib = _get_library()

        result = ctypes.c_int64()
        values = ctypes.ARRAY(ctypes.c_int32, len(label))()
        for i, v in enumerate(label):
            values[i] = ctypes.c_int32(v)

        lib.eqs_labels_position(
            self._eqs_labels_t,
            values,
            len(label),
            result,
        )

        if result.value >= 0:
            return result.value
        else:
            return None

    def asarray(self):
        """Get a view of these ``Labels`` as a raw 2D array of integers"""
        return self.view(dtype=np.int32).reshape(self.shape[0], -1)

    def __contains__(self, label):
        return self.position(label) is not None


def _eqs_labels_view(array):
    """Create a new eqs_label_t where the values are a view inside the array"""
    labels = eqs_labels_t()
    names = ctypes.ARRAY(ctypes.c_char_p, len(array.names))()
    for i, n in enumerate(array.names):
        names[i] = n.encode("utf8")

    labels.internal_ptr_ = None
    labels.names = names
    labels.size = len(array.names)

    # We need to make sure the data is C-contiguous to take a pointer to it
    contiguous = np.ascontiguousarray(array)
    ptr = contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    labels.values = ptr
    labels.count = array.shape[0]

    return labels


def _is_namedtuple(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def _print_labels(Labels: Labels, header: str, print_limit=10) -> str:
    """
    Utility function to print a Label in a table-like format, with the number
    aligned under names of the corresponding column.

    If ``len(Labels) > print_limit`` it will print only the first three columns
    and the last three.

    The result will look like

    .. code-block::

        <header>: ['name_1', 'name_2', 'name_3']
                     v11       v12       v13
                     v21       v22       v23
                     v31       v32       v33

    :param Labels: the input label
    :param header: header to place before the names of the columns
    :param print_limit: the maximum number of line you want to print without skipping
    """

    if print_limit < 6:
        print_limit = 6

    ln = len(Labels)
    width = []
    s = header + ": ["
    for i in Labels.names:
        s += "'{}' ".format(i)
        width.append(len(i) + 3)
    if ln > 0:
        s = s[:-1]  # cancel last " "
    s += "]\n"
    header_len = len(header)
    prev = header_len + 3
    if ln <= print_limit:
        for ik in Labels:
            for iw, i in zip(width, ik):
                s += _make_padding(value=i, width=iw, prev=prev)
                prev = iw // 2
            s += "\n"
            prev = header_len + 3
    else:
        for ik in Labels[:3]:
            for iw, i in zip(width, ik):
                s += _make_padding(value=i, width=iw, prev=prev)
                prev = iw // 2
            s += "\n"
            prev = header_len + 3
        s += "...\n".rjust(prev + width[0] // 2)
        for ik in Labels[-3:]:
            for iw, i in zip(width, ik):
                s += _make_padding(value=i, width=iw, prev=prev)
                prev = iw // 2
            s += "\n"
            prev = header_len + 3
    return s[:-1]


def _make_padding(value, width: int, prev=0):
    """
    Utility to make the padding in the output string

    :param value: value to write
    :param width: width of the name of that column
    :param prev: additional padding
    """
    pad = prev + width // 2
    return ("{:>" + str(pad) + "}").format(value)
