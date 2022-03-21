from collections import namedtuple
from typing import List
import numpy as np
import ctypes

from ._c_api import aml_labels_t


class Labels(np.ndarray):
    """
    A set of labels used to carry metadata associated with a descriptor.

    This is similar to a list of ``n_entries`` named tuples, but stored as a 2D
    array of shape ``(n_entries, n_variables)``, with a set of names associated
    with the columns of this array (often called *variables*). Each row/entry in
    this array is unique, and they are often (but not always) sorted in
    lexicographic order.

    In Python, :py:class:`Labels` are implemented as a wrapper around a 2D
    ``numpy.ndarray`` with a custom ``dtype`` allowing direct access to the
    different columns:

    .. code-block:: python

        labels = Labels(
            names=["structure", "atom", "center_species"]
            values=...
        )

        # access all values in a column by name
        structures = labels["structures"]

        # multiple columns at once
        data = labels[["structures", "center_species"]]

        # we can still use all the usual numpy operations
        unique_structures = np.unique(labels["structures"])
    """

    def __new__(cls, names: List[str], values: np.ndarray):
        """
        :param names: names of the variables in the new labels
        :param values: values of the variables, this needs to be a 2D array of
            ``np.int32`` values
        """
        if not isinstance(values, np.ndarray):
            raise ValueError("values parameter must be a numpy ndarray")

        if len(values.shape) != 2:
            raise ValueError("values parameter must be a 2D array")

        names = tuple(str(n) for n in names)

        if len(names) != values.shape[1]:
            raise ValueError(
                "names parameter must have an entry for each column of the array"
            )

        if values.dtype != np.int32:
            raise ValueError("values parameter must be an array of 32-bit integers")

        values = np.ascontiguousarray(values)
        dtype = [(name, np.int32) for name in names]

        if values.shape[1] != 0:
            values = values.view(dtype=dtype).reshape((values.shape[0],))

            if len(np.unique(values)) != len(values):
                raise ValueError("values in Labels must be unique")

        obj = values.view(cls)

        # keep a reference to the parent object (if any) to prevent it from
        # beeing garbage-collected when the Labels are a view inside memory
        # owned by the parent
        obj._parent = None

        return obj

    def __array_finalize__(self, obj):
        if hasattr(obj, "_parent"):
            self._parent = obj._parent
        else:
            self._parent = None

    @property
    def names(self) -> List[str]:
        """Names of the columns/variables used for these labels"""
        return self.dtype.names or []

    @staticmethod
    def single() -> "Labels":
        """
        Get the labels to use when there is no relevant metadata and only one
        entry in the corresponding dimension (i.e. scalar component labels, or
        sparse label when a descriptor contains a single block).
        """
        return Labels(names=["_"], values=np.zeros(shape=(1, 1), dtype=np.int32))

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

    def _as_aml_labels_t(self):
        """transform these labels into aml_labels_t"""
        aml_labels = aml_labels_t()

        names = ctypes.ARRAY(ctypes.c_char_p, len(self.names))()
        for i, n in enumerate(self.names):
            names[i] = n.encode("utf8")

        aml_labels.names = names
        aml_labels.size = len(names)
        aml_labels.values = self.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        aml_labels.count = self.shape[0]

        return aml_labels

    @staticmethod
    def _from_aml_labels_t(labels, parent):
        """
        Convert an aml_labels_t into a Labels instance. The Labels is only a
        view inside the aml_labels_t memory, so one can use the parent parameter
        to ensure a parent object is kept alive for as long as the Labels live.
        """
        names = []
        for i in range(labels.size):
            names.append(labels.names[i].decode("utf8"))

        if labels.count != 0:
            shape = (labels.count, labels.size)
            values = _ptr_to_ndarray(ptr=labels.values, shape=shape, dtype=np.int32)
            values.flags.writeable = False
            labels = Labels(names, values)
            labels._parent = parent
        else:
            return Labels(
                names=names,
                values=np.empty(shape=(0, len(names)), dtype=np.int32),
            )


def _is_namedtuple(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def _ptr_to_ndarray(ptr, shape, dtype):
    assert len(shape) == 2
    assert shape[1] != 0 and ptr is not None
    array = np.ctypeslib.as_array(ptr, shape=shape)
    assert array.dtype == dtype
    return array
