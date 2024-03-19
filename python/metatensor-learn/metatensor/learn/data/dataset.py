"""
Module for defining a Dataset or IndexedDataset.
"""

from collections import namedtuple
from typing import List, NamedTuple, Optional

import torch


class _BaseDataset(torch.utils.data.Dataset):
    """
    Defines the base private class for a metatensor-learn dataset.
    """

    def __init__(self, arg_name: str, size: Optional[int] = None, **kwargs):
        super(_BaseDataset, self).__init__()

        field_names = []
        fields = []
        field_sizes = []

        # Check type consistency in kwarg fields
        for name, field in kwargs.items():
            if isinstance(field, list):
                if not all([isinstance(field[0], type(f)) for f in field]):
                    raise TypeError(
                        f"Data field {name} must be a list of the same type"
                    )
                field_sizes.append(len(field))
            else:
                if not callable(field):
                    raise TypeError(
                        f"Data field must be a List or Callable, not {type(field)}"
                    )
            field_names.append(name)
            fields.append(field)

        # Check size consistency
        if size is None:
            if len(field_sizes) == 0:
                raise ValueError(
                    "If passing all data fields as callables, argument "
                    f"'{arg_name}' must also be provided."
                )
            else:
                if not all([s == field_sizes[0] for s in field_sizes]):
                    raise ValueError(
                        "Number of samples inconsistent between data fields: "
                        f"{field_sizes}"
                    )
                size = field_sizes[0]
        else:
            if len(field_sizes) > 0:
                if not all([s == size for s in field_sizes]):
                    raise ValueError(
                        f"Number of samples inconsistent between argument '{arg_name}' "
                        f"({size}) and data fields in kwargs: ({field_sizes})"
                    )

        # Define an internal numeric index list as a range from 0 to size
        self._indices = list(range(size))
        self._field_names = field_names
        self._data = {name: kwargs[name] for name in self._field_names}
        self._size = size

    def __len__(self) -> int:
        """
        Returns the length (i.e. number of samples) of the dataset
        """
        return self._size

    def __iter__(self):
        """
        Returns an iterator over the dataset.
        """
        for idx in self._indices:
            yield self[idx]


class Dataset(_BaseDataset):
    """
    Defines a PyTorch compatible :py:class:`torch.data.Dataset` class for various named
    data fields.

    The data fields are specified as keyword arguments to the constructor, where the
    keyword is the name of the field, and the value is either a list of data objects or
    a callable.

    ``size`` specifies the number of samples in the dataset. This only needs to be
    passed if all data fields are passed as callables. Otherwise, the size is inferred
    from the length of the data fields passed as lists.

    Every sample in the dataset is assigned a numeric ID from 0 to ``size - 1``. This ID
    can be used to access the corresponding sample. For instance, ``dataset[0]`` returns
    a named tuple of all data fields for the first sample in the dataset.

    A data field kwarg passed as a list must be comprised of the same type of object,
    and its length must be consistent with the ``size`` argument (if specified) and the
    length of all other data fields passed as lists.

    Otherwise a data field kwarg passed as a callable must take a single argument
    corresponding to the numeric sample ID and return the data object for that sample
    ID. This data field for a given sample is then only lazily loaded into memory when
    the :py:meth:`Dataset.__getitem__` method is called.

    :param size: Optional, an integer indicating the size of the dataset, i.e. the
        number of samples. This only needs to be specified if all data
    :param kwargs: List or Callable. Keyword arguments specifying the data fields for
        the dataset.
    """

    def __init__(self, size: Optional[int] = None, **kwargs):
        if "sample_id" in kwargs:
            raise ValueError(
                "Keyword argument 'sample_id' is not accepted by Dataset."
                " For an indexed dataset, use the IndexedDataset class"
                " instead."
            )

        super(Dataset, self).__init__(arg_name="size", size=size, **kwargs)

    def __getitem__(self, idx: int) -> NamedTuple:
        """
        Returns the data for each field corresponding to the internal index ``idx``.

        Each item can be accessed with ``self[idx]``. Returned is a named tuple with
        fields corresponding to those passed (in order) to the constructor upon class
        initialization.
        """
        if idx not in self._indices:
            raise ValueError(f"Index {idx} not in dataset")
        names = self._field_names
        sample_data = []
        for name in self._field_names:
            if callable(self._data[name]):  # lazy load
                try:
                    sample_data.append(self._data[name](idx))
                except Exception as e:
                    raise IOError(
                        f"Error loading data field '{name}' at numeric"
                        f" sample index {idx}: {e}"
                    )
            else:
                assert isinstance(self._data[name], list)
                sample_data.append(self._data[name][idx])

        return namedtuple("Sample", names)(*sample_data)


class IndexedDataset(_BaseDataset):
    """
    Defines a PyTorch compatible :py:class:`torch.data.Dataset` class for various named
    data fields, with sample indexed by a list of unique sample IDs.

    The data fields are specified as keyword arguments to the constructor, where the
    keyword is the name of the field, and the value is either a list of data objects or
    a callable.

    ``sample_id`` must be a unique list of any hashable object. Each respective sample
    ID is assigned an internal numeric index from 0 to ``len(sample_id) - 1``. This is
    used to internally index the dataset, and can be used to access a given sample. For
    instance, ``dataset[0]`` returns a named tuple of all data fields for the first
    sample in the dataset, i.e. the one with unique sample ID at ``sample_id[0]``. In
    order to access a sample by its ID, use the :py:meth:`IndexedDataset.get_sample`
    method.

    A data field kwarg passed as a list must be comprised of the same type of object,
    and its length must be consistent with the length of ``sample_id`` and the length of
    all other data fields passed as lists.

    Otherwise a data field kwarg passed as a callable must take a single argument
    corresponding to the unique sample ID (i.e. those passed in ``sample_id``) and
    return the data object for that sample ID. This data field for a given sample is
    then only lazily loaded into memory when :py:meth:`IndexedDataset.__getitem__` or
    :py:meth:`IndexedDataset.get_sample` methods are called.

    :param sample_id: A list of unique IDs for each sample in the dataset.
    :param kwargs: Keyword arguments specifying the data fields for the dataset.
    """

    def __init__(self, sample_id: List, **kwargs):
        if "size" in kwargs:
            raise ValueError(
                "Keyword argument 'size' is not accepted by IndexedDataset."
                " For a dataset defined on size rather than explicit sample IDs,"
                " use the Dataset class instead."
            )
        if len(set(sample_id)) != len(sample_id):
            raise ValueError("Sample IDs must be unique. Found duplicate sample IDs.")

        super(IndexedDataset, self).__init__(
            arg_name="sample_id", size=len(sample_id), **kwargs
        )
        self._sample_id = sample_id
        self._sample_id_to_idx = {smpl_id: idx for idx, smpl_id in enumerate(sample_id)}

    def __getitem__(self, idx: int) -> NamedTuple:
        """
        Returns the data for each field corresponding to the internal index ``idx``.

        Each item can be accessed with ``self[idx]``. Returned is a named tuple, whose
        first field is the sample ID, and the remaining fields correspond those passed
        (in order) to the constructor upon class initialization.
        """
        if idx not in self._indices:
            raise ValueError(f"Index {idx} not in dataset")
        names = ["sample_id"] + self._field_names
        sample_data = [self._sample_id[idx]]
        for name in self._field_names:
            if callable(self._data[name]):  # lazy load using sample ID
                try:
                    sample_data.append(self._data[name](self._sample_id[idx]))
                except Exception as e:
                    raise IOError(
                        f"Error loading data field '{name}' for sample"
                        f" ID {self._sample_id[idx]}: {e}"
                    )
            else:
                assert isinstance(self._data[name], list)
                sample_data.append(self._data[name][idx])

        return namedtuple("Sample", names)(*sample_data)

    def get_sample(self, sample_id) -> NamedTuple:
        """
        Returns a named tuple for the sample corresponding to the given ``sample_id``.
        """
        if sample_id not in self._sample_id_to_idx:
            raise ValueError(f"Sample ID '{sample_id}' not in dataset")
        return self[self._sample_id_to_idx[sample_id]]
