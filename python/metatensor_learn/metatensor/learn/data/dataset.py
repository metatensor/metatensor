from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

import torch

from ._namedtuple import namedtuple


class _BaseDataset(torch.utils.data.Dataset):
    """
    Defines the base private class for a metatensor-learn dataset.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        size_arg_name: str,
        size: Optional[int] = None,
    ):
        super(_BaseDataset, self).__init__()

        field_names = []
        fields = []
        field_sizes = []

        # Check type consistency in kwarg fields
        for name, field in data.items():
            if isinstance(field, list):
                if name != "sample_id":
                    # sample_id is allowed to used different types for different entries
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
                    f"'{size_arg_name}' must also be provided."
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
                        "Number of samples inconsistent between argument "
                        f"'{size_arg_name}' ({size}) and data fields: ({field_sizes})"
                    )

        self._field_names = field_names
        self._data = data
        self._size = size

        self._sample_class = namedtuple("Sample", self._field_names)

    def __len__(self) -> int:
        """
        Returns the length (i.e. number of samples) of the dataset
        """
        return self._size

    def __iter__(self):
        """
        Returns an iterator over the dataset.
        """
        for idx in range(len(self)):
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

    >>> # create a Dataset with only lists
    >>> dataset = Dataset(num=[1, 2, 3], string=["a", "b", "c"])
    >>> dataset[0]
    Sample(num=1, string='a')
    >>> dataset[2]
    Sample(num=3, string='c')

    >>> # create a Dataset with callables for lazy loading of data
    >>> def call_me(sample_id: int):
    ...     # this could also read from a file
    ...     return f"compute something with sample {sample_id}"
    >>> dataset = Dataset(num=[1, 2, 3], call_me=call_me)
    >>> dataset[0]
    Sample(num=1, call_me='compute something with sample 0')
    >>> dataset[2]
    Sample(num=3, call_me='compute something with sample 2')

    >>> # iterating over a dataset
    >>> for num, called in dataset:
    ...     print(num, " -- ", called)
    1  --  compute something with sample 0
    2  --  compute something with sample 1
    3  --  compute something with sample 2
    >>> for sample in dataset:
    ...     print(sample.num, " -- ", sample.call_me)
    1  --  compute something with sample 0
    2  --  compute something with sample 1
    3  --  compute something with sample 2
    """

    def __init__(self, size: Optional[int] = None, **kwargs):
        """
        :param size: Optional, an integer indicating the size of the dataset, i.e. the
            number of samples. This only needs to be specified if all the fields are
            callable.
        :param kwargs: List or Callable. Keyword arguments specifying the data fields
            for the dataset.
        """
        if "sample_id" in kwargs:
            raise ValueError(
                "Keyword argument 'sample_id' is not accepted by Dataset."
                " For an indexed dataset, use the IndexedDataset class"
                " instead."
            )

        super().__init__(data=kwargs, size_arg_name="size", size=size)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Union[List, Callable]],
        size: Optional[int] = None,
    ) -> "Dataset":
        """
        Create a :py:class:`Dataset` from the given ``data``. This function behave like
        ``Dataset(size=size, **data)``, but allows to use names for the different fields
        that would not be valid in the main constructor.

        >>> dataset = Dataset.from_dict(
        ...     {
        ...         "valid": [0, 0, 0],
        ...         "with space": [-1, 2, -3],
        ...         "with/slash": ["a", "b", "c"],
        ...     }
        ... )
        >>> sample = dataset[1]
        >>> sample
        Sample(valid=0, 'with space'=2, 'with/slash'='b')
        >>> # fields for which the name is a valid identifier can be accessed as usual
        >>> sample.valid
        0
        >>> # fields which are not valid identifiers can be accessed like this
        >>> sample["with space"]
        2
        >>> sample["with/slash"]
        'b'

        :param data: Dictionary of List or Callable containing the data. This will
            behave as if all the entries of the dictionary where passed as keyword
            arguments to ``__init__``.
        :param size: Optional, an integer indicating the size of the dataset, i.e. the
            number of samples. This only needs to be specified if all the fields are
            callable.
        """
        if "sample_id" in data:
            raise ValueError(
                "'sample_id' is not accepted by Dataset. For an indexed dataset, "
                "use the IndexedDataset class instead."
            )

        dataset = cls.__new__(cls)
        _BaseDataset.__init__(self=dataset, data=data, size_arg_name="size", size=size)
        return dataset

    def __getitem__(self, idx: int) -> NamedTuple:
        """
        Returns the data for each field corresponding to the internal index ``idx``.

        Each item can be accessed with ``self[idx]``. Returned is a named tuple with
        fields corresponding to those passed (in order) to the constructor upon class
        initialization.
        """
        idx = int(idx)
        if idx >= len(self) or idx < 0:
            raise ValueError(f"Index {idx} not in dataset")

        sample_data = []
        for name in self._field_names:
            if callable(self._data[name]):  # lazy load
                try:
                    sample_data.append(self._data[name](idx))
                except Exception as e:
                    raise ValueError(
                        f"Error loading data field '{name}' at sample index {idx}"
                    ) from e
            else:
                assert isinstance(self._data[name], list)
                sample_data.append(self._data[name][idx])

        return self._sample_class(*sample_data)


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

    >>> # create an IndexedDataset with lists
    >>> dataset = IndexedDataset(sample_id=["cat", "bird", "dog"], y=[11, 22, 33])
    >>> dataset[0]
    Sample(sample_id='cat', y=11)
    >>> dataset[2]
    Sample(sample_id='dog', y=33)

    >>> # create an IndexedDataset with callables for lazy loading of data
    >>> def call_me(sample_id: int):
    ...     # this could also read from a file
    ...     return f"compute something with sample {sample_id}"
    >>> dataset = IndexedDataset(sample_id=["cat", "bird", "dog"], call_me=call_me)
    >>> dataset[0]
    Sample(sample_id='cat', call_me='compute something with sample cat')
    >>> dataset[2]
    Sample(sample_id='dog', call_me='compute something with sample dog')
    >>> dataset.get_sample("bird")
    Sample(sample_id='bird', call_me='compute something with sample bird')

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
            raise ValueError("Sample IDs must be unique, found some duplicate")

        data = {"sample_id": sample_id}
        data.update(kwargs)
        super().__init__(data=data, size_arg_name="sample_id", size=len(sample_id))

        self._sample_id = sample_id
        self._sample_id_to_idx = {sample: i for i, sample in enumerate(sample_id)}

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        sample_id: List,
    ) -> "IndexedDataset":
        """
        Create a :py:class:`IndexedDataset` from the given ``data``. This function
        behave like ``IndexedDataset(sample_id=sample_id, **data)``, but allows to use
        names for the different fields that would not be valid in the main constructor.

        >>> dataset = IndexedDataset.from_dict(
        ...     {
        ...         "valid": [0, 0, 0],
        ...         "with space": [-1, 2, -3],
        ...         "with/slash": ["a", "b", "c"],
        ...     },
        ...     sample_id=[11, 22, 33],
        ... )
        >>> sample = dataset[1]
        >>> sample
        Sample(sample_id=22, valid=0, 'with space'=2, 'with/slash'='b')
        >>> # fields for which the name is a valid identifier can be accessed as usual
        >>> sample.valid
        0
        >>> # fields which are not valid identifiers can be accessed like this
        >>> sample["with space"]
        2
        >>> sample["with/slash"]
        'b'

        :param data: Dictionary of List or Callable containing the data. This will
            behave as if all the entries of the dictionary where passed as keyword
            arguments to ``__init__``.
        :param sample_id: A list of unique IDs for each sample in the dataset.
        """
        if "size" in data:
            raise ValueError(
                "'size' is not accepted by IndexedDataset. For a dataset defined on "
                "size rather than explicit sample IDs, use the Dataset class instead."
            )

        if len(set(sample_id)) != len(sample_id):
            raise ValueError("Sample IDs must be unique, found some duplicate")

        clean_data = {"sample_id": sample_id}
        clean_data.update(data)

        dataset = cls.__new__(cls)
        _BaseDataset.__init__(
            self=dataset,
            data=clean_data,
            size_arg_name="sample_id",
            size=len(sample_id),
        )

        dataset._sample_id = sample_id
        dataset._sample_id_to_idx = {sample: i for i, sample in enumerate(sample_id)}
        return dataset

    def __getitem__(self, idx: int) -> NamedTuple:
        """
        Returns the data for each field corresponding to the internal index ``idx``.

        Each item can be accessed with ``self[idx]``. Returned is a named tuple, whose
        first field is the sample ID, and the remaining fields correspond those passed
        (in order) to the constructor upon class initialization.
        """
        idx = int(idx)
        if idx >= len(self) or idx < 0:
            raise ValueError(f"Index {idx} not in dataset")

        sample_id = self._sample_id[idx]

        sample_data = []
        for name in self._field_names:
            if name == "sample_id":
                sample_data.append(sample_id)
            elif callable(self._data[name]):  # lazy load using sample ID
                try:
                    sample_data.append(self._data[name](sample_id))
                except Exception as e:
                    raise ValueError(
                        f"Error loading data field '{name}' for sample '{sample_id}'"
                    ) from e
            else:
                assert isinstance(self._data[name], list)
                sample_data.append(self._data[name][idx])

        return self._sample_class(*sample_data)

    def get_sample(self, sample_id) -> NamedTuple:
        """
        Returns a named tuple for the sample corresponding to the given ``sample_id``.
        """
        if sample_id not in self._sample_id_to_idx:
            raise ValueError(f"Sample ID '{sample_id}' not in dataset")
        return self[self._sample_id_to_idx[sample_id]]
