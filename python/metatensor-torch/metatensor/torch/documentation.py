from typing import Dict, List, Optional, Tuple, Union, overload

import torch


# These classes do not contain the actual code (see the C++ code in the root
# metatensor-torch folder for that), but instead include the documentation for
# these classes in a way compatible with Sphinx.

StrSequence = Union[str, List[str], Tuple[str, ...]]


class LabelsEntry:
    """A single entry (i.e. row) in a set of :py:class:`Labels`.

    The main way to create a :py:class:`LabelsEntry` is to index a
    :py:class:`Labels` or iterate over them.

    >>> from metatensor.torch import Labels
    >>> labels = Labels(
    ...     names=["structure", "atom", "species_center"],
    ...     values=torch.tensor([(0, 1, 8), (0, 2, 1), (0, 5, 1)]),
    ... )
    >>> entry = labels[0]  # or labels.entry(0)
    >>> entry.names
    ['structure', 'atom', 'species_center']
    >>> entry.values
    tensor([0, 1, 8], dtype=torch.int32)

    .. warning::

        Due to limitations in TorchScript, :py:class:`LabelsEntry`
        implementation of ``__hash__`` will use the default Python one,
        returning the ``id()`` of the object. If you want to use
        :py:class:`LabelsEntry` as keys in a dictionary, convert them to tuple
        first (``tuple(entry)``) --- or to string (``str(entry)``) since
        TorchScript does not support tuple as dictionary keys anyway.
    """

    @property
    def names(self) -> List[str]:
        """names of the dimensions for this Labels entry"""

    @property
    def values(self) -> torch.Tensor:
        """
        Values associated with each dimensions of this :py:class:`LabelsEntry`, stored
        as 32-bit integers.

        .. warning::

            The ``values`` should be treated as immutable/read-only (we would like to
            enforce this automatically, but PyTorch can not mark a
            :py:class:`torch.Tensor` as immutable)

            Any modification to this tensor can break the underlying data structure, or
            make it out of sync with the ``values``.
        """

    def print(self) -> str:
        """
        print this entry as a named tuple (i.e. ``(key_1=value_1, key_2=value_2)``)
        """

    def __len__(self) -> int:
        """number of dimensions in this labels entry"""

    def __getitem__(self, dimension: Union[str, int]) -> int:
        """get the value associated with the dimension in this entry"""

    def __eq__(self, other: "LabelsEntry") -> bool:
        """
        check if ``self`` and ``other`` are equal (same dimensions/names and
        same values)
        """

    def __ne__(self, other: "LabelsEntry") -> bool:
        """
        check if ``self`` and ``other`` are not equal (different dimensions/names or
        different values)
        """


class Labels:
    """
    A set of labels carrying metadata associated with a :py:class:`TensorMap`.

    The metadata can be though as a list of tuples, where each value in the
    tuple also has an associated dimension name. In practice, the dimensions
    ``names`` are stored separately from the ``values``, and the values are in a
    2-dimensional array integers with the shape ``(n_entries, n_dimensions)``.
    Each row/entry in this array is unique, and they are often (but not always)
    sorted in lexicographic order.

    ..  seealso::

        The pure Python version of this class :py:class:`metatensor.Labels`,
        and the :ref:`differences between TorchScript and Python API for
        metatensor <python-vs-torch>`.

    >>> from metatensor.torch import Labels
    >>> labels = Labels(
    ...     names=["structure", "atom", "species_center"],
    ...     values=torch.tensor([(0, 1, 8), (0, 2, 1), (0, 5, 1)]),
    ... )
    >>> print(labels)
    Labels(
        structure  atom  species_center
            0       1          8
            0       2          1
            0       5          1
    )
    >>> labels.names
    ['structure', 'atom', 'species_center']
    >>> labels.values
    tensor([[0, 1, 8],
            [0, 2, 1],
            [0, 5, 1]], dtype=torch.int32)


    It is possible to create a view inside a :py:class:`Labels`, selecting only
    a subset of columns/dimensions:

    >>> # single dimension
    >>> view = labels.view("atom")
    >>> view.names
    ['atom']
    >>> view.values
    tensor([[1],
            [2],
            [5]], dtype=torch.int32)
    >>> # multiple dimensions
    >>> view = labels.view(["atom", "structure"])
    >>> view.names
    ['atom', 'structure']
    >>> view.values
    tensor([[1, 0],
            [2, 0],
            [5, 0]], dtype=torch.int32)
    >>> view.is_view()
    True
    >>> # we can convert a view back to a full, owned Labels
    >>> owned_labels = view.to_owned()
    >>> owned_labels.is_view()
    False

    One can also iterate over labels entries, or directly index the :py:class:`Labels`
    to get a specific entry

    >>> entry = labels[0]  # or labels.entry(0)
    >>> entry.names
    ['structure', 'atom', 'species_center']
    >>> entry.values
    tensor([0, 1, 8], dtype=torch.int32)
    >>> for entry in labels:
    ...     print(entry)
    ...
    LabelsEntry(structure=0, atom=1, species_center=8)
    LabelsEntry(structure=0, atom=2, species_center=1)
    LabelsEntry(structure=0, atom=5, species_center=1)

    Or get all the values associated with a given dimension/column name

    >>> labels.column("atom")
    tensor([1, 2, 5], dtype=torch.int32)
    >>> labels["atom"]  # alternative syntax for the above
    tensor([1, 2, 5], dtype=torch.int32)

    Labels can be checked for equality:

    >>> owned_labels == labels
    False
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

    def __init__(self, names: StrSequence, values: torch.Tensor):
        """
        :param names: names of the dimensions in the new labels. A single string
                      is transformed into a list with one element, i.e.
                      ``names="a"`` is the same as ``names=["a"]``.

        :param values: values of the labels, this needs to be a 2-dimensional
                       array of integers.
        """

    @property
    def names(self) -> List[str]:
        """names of the dimensions for these :py:class:`Labels`"""

    @property
    def values(self) -> torch.Tensor:
        """
        Values associated with each dimensions of the :py:class:`Labels`, stored
        as 2-dimensional tensor of 32-bit integers.

        .. warning::

            The ``values`` should be treated as immutable/read-only (we would like to
            enforce this automatically, but PyTorch can not mark a
            :py:class:`torch.Tensor` as immutable)

            Any modification to this tensor can break the underlying data structure, or
            make it out of sync with the ``values``.
        """

    @staticmethod
    def single() -> "Labels":
        """
        Create :py:class:`Labels` to use when there is no relevant metadata and
        only one entry in the corresponding dimension (e.g. keys when a tensor
        map contains a single block).
        """

    @staticmethod
    def empty(names: StrSequence) -> "Labels":
        """
        Create :py:class:`Labels` with given ``names`` but no values.

        :param names: names of the dimensions in the new labels. A single string
                      is transformed into a list with one element, i.e.
                      ``names="a"`` is the same as ``names=["a"]``.
        """

    @staticmethod
    def range(name: str, end: int) -> "Labels":
        """
        Create :py:class:`Labels` with a single dimension using the given
        ``name`` and values in the ``[0, end)`` range.

        :param name: name of the single dimension in the new labels.
        :param end: end of the range for labels

        >>> from metatensor.torch import Labels
        >>> labels = Labels.range("dummy", 7)
        >>> labels.names
        ['dummy']
        >>> labels.values
        tensor([[0],
                [1],
                [2],
                [3],
                [4],
                [5],
                [6]], dtype=torch.int32)
        """

    def __len__(self) -> int:
        """number of entries in these labels"""

    @overload
    def __getitem__(self, dimension: str) -> torch.Tensor:
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

        See also :py:func:`Labels.view` to extract the values associated with multiple
        columns/dimensions.
        """

    def __contains__(
        self, entry: Union[LabelsEntry, torch.Tensor, List[int], Tuple[int, ...]]
    ) -> bool:
        """check if these :py:class:`Labels` contain the given ``entry``"""

    def __eq__(self, other: "Labels") -> bool:
        """
        check if two set of labels are equal (same dimension names and same
        values)
        """

    def __ne__(self, other: "Labels") -> bool:
        """
        check if two set of labels are not equal (different dimension names or
        different values)
        """

    def append(self, name: str, values: torch.Tensor) -> "Labels":
        """Append a new dimension to the end of the :py:class:`Labels`.

        :param name: name of the new dimension
        :param values: 1D array of values for the new dimension

        >>> import torch
        >>> from metatensor.torch import Labels
        >>> label = Labels("foo", torch.tensor([[42]]))
        >>> print(label)
        Labels(
            foo
            42
        )
        >>> print(label.append(name="bar", values=torch.tensor([10])))
        Labels(
            foo  bar
            42   10
        )
        """

    def insert(self, index: int, name: str, values: torch.Tensor) -> "Labels":
        """Insert a new dimension before ``index`` in the :py:class:`Labels`.

        :param index: index before the new dimension is inserted
        :param name: name of the new dimension
        :param values: 1D array of values for the new dimension

        >>> import torch
        >>> from metatensor.torch import Labels
        >>> label = Labels("foo", torch.tensor([[42]]))
        >>> print(label)
        Labels(
            foo
            42
        )
        >>> print(label.insert(0, name="bar", values=torch.tensor([10])))
        Labels(
            bar  foo
            10   42
        )
        """

    def permute(self, dimensions_indexes: List[int]) -> "Labels":
        """Permute dimensions according to ``dimensions_indexes`` in the
        :py:class:`Labels`.

        :param dimensions_indexes: desired ordering of the dimensions
        :raises ValueError: if length of ``dimensions_indexes`` does not match the
            Labels length
        :raises ValueError: if duplicate values are present in ``dimensions_indexes``

        >>> import torch
        >>> from metatensor.torch import Labels
        >>> label = Labels(["foo", "bar", "baz"], torch.tensor([[42, 10, 3]]))
        >>> print(label)
        Labels(
            foo  bar  baz
            42   10    3
        )
        >>> print(label.permute([2, 0, 1]))
        Labels(
            baz  foo  bar
             3   42   10
        )
        """

    def remove(self, name: str) -> "Labels":
        """Remove ``name`` from the dimensions of the :py:class:`Labels`.

        Removal can only be performed if the resulting :py:class:`Labels` instance will
        be unique.

        :param name: name to be removed
        :raises ValueError: if the name is not present.

        >>> import torch
        >>> from metatensor.torch import Labels
        >>> label = Labels(["foo", "bar"], torch.tensor([[42, 10]]))
        >>> print(label)
        Labels(
            foo  bar
            42   10
        )
        >>> print(label.remove(name="bar"))
        Labels(
            foo
            42
        )

        If the new :py:class:`Labels` is not unique an error is raised.

        >>> label = Labels(["foo", "bar"], torch.tensor([[42, 10], [42, 11]]))
        >>> print(label)
        Labels(
            foo  bar
            42   10
            42   11
        )
        >>> try:
        ...     label.remove(name="bar")
        ... except RuntimeError as e:
        ...     print(e)
        ...
        invalid parameter: can not have the same label value multiple time: [42] is already present at position 0
        """  # noqa E501

    def rename(self, old: str, new: str) -> "Labels":
        """Rename the ``old`` dimension to ``new`` in the :py:class:`Labels`.

        :param old: name to be replaced
        :param new: name after the replacement
        :raises ValueError: if old is not present.

        >>> import torch
        >>> from metatensor.torch import Labels
        >>> label = Labels("foo", torch.tensor([[42]]))
        >>> print(label)
        Labels(
            foo
            42
        )
        >>> print(label.rename("foo", "bar"))
        Labels(
            bar
            42
        )
        """

    @overload
    def to(self, str) -> "Labels":
        """move the values for these Labels to the given ``device``"""

    @overload
    def to(self, device) -> "Labels":
        """move the values for these Labels to the given ``device``"""

    def position(
        self, entry: Union[LabelsEntry, torch.Tensor, List[int], Tuple[int, ...]]
    ) -> Optional[int]:
        """
        Get the position of the given ``entry`` in this set of
        :py:class:`Labels`, or ``None`` if the entry is not present in the
        labels.
        """

    def union(self, other: "Labels") -> "Labels":
        """
        Take the union of these :py:class:`Labels` with ``other``.

        If you want to know where entries in ``self`` and ``other`` ends up in the
        union, you can use :py:meth:`Labels.union_and_mapping`.
        """

    def union_and_mapping(
        self, other: "Labels"
    ) -> Tuple["Labels", torch.Tensor, torch.Tensor]:
        """
        Take the union of these :py:class:`Labels` with ``other``.

        This function also returns the position in the union where each entry of the
        input :py:class::`Labels` ended up.

        :return: Tuple containing the union, a :py:class:`torch.Tensor` containing the
            position in the union of the entries from ``self``, and a
            :py:class:`torch.Tensor` containing the position in the union of the
            entries from ``other``.
        """

    def intersection(self, other: "Labels") -> "Labels":
        """
        Take the intersection of these :py:class:`Labels` with ``other``.

        If you want to know where entries in ``self`` and ``other`` ends up in the
        intersection, you can use :py:meth:`Labels.intersection_and_mapping`.
        """

    def intersection_and_mapping(
        self, other: "Labels"
    ) -> Tuple["Labels", torch.Tensor, torch.Tensor]:
        """
        Take the intersection of these :py:class:`Labels` with ``other``.

        This function also returns the position in the intersection where each entry of
        the input :py:class::`Labels` ended up.

        :return: Tuple containing the intersection, a :py:class:`torch.Tensor`
            containing the position in the intersection of the entries from ``self``,
            and a :py:class:`torch.Tensor` containing the position in the intersection
            of the entries from ``other``. If entries in ``self`` or ``other`` are not
            used in the output, the mapping for them is set to ``-1``.
        """

    def print(self, max_entries: int, indent: int) -> str:
        """print these :py:class:`Labels` to a string

        :param max_entries: how many entries to print, use ``-1`` to print everything
        :param indent: indent the output by ``indent`` spaces
        """

    def entry(self, index: int) -> LabelsEntry:
        """get a single entry in these labels, see also :py:func:`Labels.__getitem__`"""

    def column(self, dimension: str) -> torch.Tensor:
        """
        Get the values associated with a single dimension in these labels (i.e. a single
        column of :py:attr:`Labels.values`) as a 1-dimensional array.

        .. seealso::

            :py:func:`Labels.__getitem__` as the main way to use this function

            :py:func:`Labels.view` to access multiple columns simultaneously
        """

    def view(self, dimensions: StrSequence) -> "Labels":
        """get a view for the specified columns in these labels, see also
        :py:func:`Labels.__getitem__`"""

    def is_view(self) -> bool:
        """are these labels a view inside another set of labels?

        A view is created with :py:func:`Labels.__getitem__` or
        :py:func:`Labels.view`, and does not implement :py:func:`Labels.position`
        or :py:func:`Labels.__contains__`.
        """

    def to_owned(self) -> "Labels":
        """convert a view to owned labels, which implement the full API"""


class TensorBlock:
    """
    Basic building block for a :py:class:`TensorMap`.

    A single block contains a n-dimensional :py:class:`torch.Tensor` of values,
    and n sets of :py:class:`Labels` (one for each dimension). The first
    dimension is the *samples* dimension, the last dimension is the *properties*
    dimension. Any intermediate dimension is called a *component* dimension.

    Samples should be used to describe *what* we are representing, while
    properties should contain information about *how* we are representing it.
    Finally, components should be used to describe vectorial or tensorial
    components of the data.

    A block can also contain gradients of the values with respect to a variety
    of parameters. In this case, each gradient is a :py:class:`TensorBlock` with
    a separate set of samples and possibly components, but which shares the same
    property labels as the original :py:class:`TensorBlock`.

    .. seealso::

        The pure Python version of this class
        :py:class:`metatensor.TensorBlock`, and the :ref:`differences
        between TorchScript and Python API for metatensor <python-vs-torch>`.
    """

    def __init__(
        self,
        values: torch.Tensor,
        samples: Labels,
        components: List[Labels],
        properties: Labels,
    ):
        """
        :param values: tensor containing the values for this block
        :param samples: labels describing the samples (first dimension of the array)
        :param components: list of labels describing the components (intermediate
            dimensions of the array). This should be an empty list for scalar/invariant
            data.
        :param properties: labels describing the properties (last dimension of the
            array)
        """

    @property
    def values(self) -> torch.Tensor:
        """get the values for this block"""

    @property
    def samples(self) -> Labels:
        """
        Get the sample :py:class:`Labels` for this block.

        The entries in these labels describe the first dimension of the
        ``values`` array.
        """

    @property
    def components(self) -> List[Labels]:
        """
        Get the component :py:class:`Labels` for this block.

        The entries in these labels describe intermediate dimensions of the
        ``values`` array.
        """

    @property
    def properties(self) -> Labels:
        """
        Get the property :py:class:`Labels` for this block.

        The entries in these labels describe the last dimension of the
        ``values`` array. The properties are guaranteed to be the same for
        values and gradients in the same block.
        """

    def copy(self) -> "TensorBlock":
        """get a deep copy of this block, including all the data and metadata"""

    def add_gradient(self, parameter: str, gradient: "TensorBlock"):
        """
        Add gradient with respect to ``parameter`` in this block.

        :param parameter:
            add gradients with respect to this ``parameter`` (e.g. ``positions``,
            ``cell``, ...)

        :param gradient:
            a :py:class:`TensorBlock` whose values contain the gradients of this
            :py:class:`TensorBlock` values with respect to ``parameter``. The labels
            of the gradient :py:class:`TensorBlock` should be organized as follows:

            - its samples must contain ``"sample"`` as the first dimension, with values
              containing the index of the corresponding samples in this
              :py:class:`TensorBlock`, and arbitrary supplementary samples dimension;
            - its components must contain at least the same components as this
              :py:class:`TensorBlock`, with any additional components coming before
              those;
            - its properties must match exactly those of this :py:class:`TensorBlock`.

        >>> import numpy as np
        >>> from metatensor.torch import TensorBlock, Labels
        >>> block = TensorBlock(
        ...     values=torch.full((3, 1, 1), 1.0),
        ...     samples=Labels(["structure"], torch.IntTensor([[0], [2], [4]])),
        ...     components=[Labels.range("component", 1)],
        ...     properties=Labels.range("property", 1),
        ... )
        >>> gradient = TensorBlock(
        ...     values=torch.full((2, 1, 1), 11.0),
        ...     samples=Labels(
        ...         names=["sample", "parameter"],
        ...         values=torch.IntTensor([[0, -2], [2, 3]]),
        ...     ),
        ...     components=[Labels.range("component", 1)],
        ...     properties=Labels.range("property", 1),
        ... )
        >>> block.add_gradient("parameter", gradient)
        >>> print(block)
        TensorBlock
            samples (3): ['structure']
            components (1): ['component']
            properties (1): ['property']
            gradients: ['parameter']
        <BLANKLINE>
        """

    def gradient(self, parameter: str) -> "TensorBlock":
        """
        Get the gradient of the block ``values``  with respect to the given
        ``parameter``.

        :param parameter: check for gradients with respect to this ``parameter``
            (e.g. ``positions``, ``cell``, ...)

        >>> from metatensor.torch import TensorBlock, Labels
        >>> block = TensorBlock(
        ...     values=torch.full((3, 1, 5), 1.0),
        ...     samples=Labels(["structure"], torch.tensor([[0], [2], [4]])),
        ...     components=[Labels.range("component", 1)],
        ...     properties=Labels.range("property", 5),
        ... )

        >>> positions_gradient = TensorBlock(
        ...     values=torch.full((2, 3, 1, 5), 11.0),
        ...     samples=Labels(["sample", "atom"], torch.tensor([[0, 2], [2, 3]])),
        ...     components=[
        ...         Labels.range("direction", 3),
        ...         Labels.range("component", 1),
        ...     ],
        ...     properties=Labels.range("property", 5),
        ... )
        >>> block.add_gradient("positions", positions_gradient)

        >>> cell_gradient = TensorBlock(
        ...     values=torch.full((2, 3, 3, 1, 5), 15.0),
        ...     samples=Labels.range("sample", 2),
        ...     components=[
        ...         Labels.range("direction_1", 3),
        ...         Labels.range("direction_2", 3),
        ...         Labels.range("component", 1),
        ...     ],
        ...     properties=Labels.range("property", 5),
        ... )
        >>> block.add_gradient("cell", cell_gradient)

        >>> positions_gradient = block.gradient("positions")
        >>> print(positions_gradient)
        Gradient TensorBlock ('positions')
            samples (2): ['sample', 'atom']
            components (3, 1): ['direction', 'component']
            properties (5): ['property']
            gradients: None
        <BLANKLINE>
        >>> cell_gradient = block.gradient("cell")
        >>> print(cell_gradient)
        Gradient TensorBlock ('cell')
            samples (2): ['sample']
            components (3, 3, 1): ['direction_1', 'direction_2', 'component']
            properties (5): ['property']
            gradients: None
        <BLANKLINE>
        """

    def gradients_list(self) -> List[str]:
        """get a list of all gradients defined in this block"""

    def has_gradient(self, parameter: str) -> bool:
        """
        Check if this block contains gradient information with respect to the
        given ``parameter``.

        :param parameter: check for gradients with respect to this ``parameter``
            (e.g. ``positions``, ``cell``, ...)
        """

    def gradients(self) -> Dict[str, "TensorBlock"]:
        """
        Get an iterator over all (parameter, gradients) pairs defined in this
        block.
        """


class TensorMap:
    """
    A TensorMap is the main user-facing class of this library, and can store any
    kind of data used in atomistic machine learning.

    A tensor map contains a list of :py:class:`TensorBlock`, each one associated
    with a key. It also provides functions to access blocks associated with a
    key or part of a key, functions to merge blocks together and in general to
    manipulate this collection of blocks.

    ..  seealso::

        The pure Python version of this class
        :py:class:`metatensor.TensorMap`, and the :ref:`differences between
        TorchScript and Python API for metatensor <python-vs-torch>`.
    """

    def __init__(self, keys: Labels, blocks: List[TensorBlock]):
        """
        :param keys: keys associated with each block
        :param blocks: set of blocks containing the actual data
        """

    @property
    def keys(self) -> Labels:
        """the set of keys labeling the blocks in this :py:class:`TensorMap`"""

    def __len__(self) -> int:
        """get the number of key/block pairs in this :py:class:`TensorMap`"""

    def __getitem__(
        self,
        selection: Union[int, Labels, LabelsEntry, Dict[str, int]],
    ) -> TensorBlock:
        """
        Get a single block with indexing syntax. This calls :py:func:`TensorMap.block`
        directly.
        """

    def copy(self) -> "TensorMap":
        """
        get a deep copy of this :py:class:`TensorMap`, including all the data
        and metadata
        """

    def items(self) -> List[Tuple[LabelsEntry, TensorBlock]]:
        """get an iterator over (key, block) pairs in this :py:class:`TensorMap`"""

    def keys_to_samples(
        self,
        keys_to_move: Union[StrSequence, Labels],
        sort_samples: bool = True,
    ) -> "TensorMap":
        """
        Merge blocks along the samples axis, adding ``keys_to_move`` to the end
        of the samples labels dimensions.

        This function will remove ``keys_to_move`` from the keys, and find all
        blocks with the same remaining keys values. It will then merge these
        blocks along the samples direction (i.e. do a *vertical* concatenation),
        adding ``keys_to_move`` to the end of the samples labels dimensions.
        The values taken by ``keys_to_move`` in the new samples labels will be
        the values of these dimensions in the merged blocks' keys.

        If ``keys_to_move`` is a set of :py:class:`Labels`, it must be empty
        (``len(keys_to_move) == 0``), and only the :py:class:`Labels.names` will
        be used.

        The order of the samples in the merged blocks is controlled by
        ``sort_samples``. If ``sort_samples`` is :py:obj:`True`, samples are
        re-ordered to keep them lexicographically sorted. Otherwise they are
        kept in the order in which they appear in the blocks.

        This function is only implemented when the blocks to merge have the same
        properties values.

        :param keys_to_move: description of the keys to move
        :param sort_samples: whether to sort the merged samples or keep them in
            the order in which they appear in the original blocks
        :return: a new :py:class:`TensorMap` with merged blocks
        """

    def keys_to_properties(
        self,
        keys_to_move: Union[StrSequence, Labels],
        sort_samples: bool = True,
    ) -> "TensorMap":
        """
        Merge blocks along the properties direction, adding ``keys_to_move`` at
        the beginning of the properties labels dimensions.

        This function will remove ``keys_to_move`` from the keys, and find all
        blocks with the same remaining keys values. Then it will merge these
        blocks along the properties direction (i.e. do an *horizontal*
        concatenation).

        If ``keys_to_move`` is given as strings, then the new property labels
        will **only** contain entries from the existing blocks. For example,
        merging a block with key ``a=0`` and properties ``p=1, 2`` with a block
        with key ``a=2`` and properties ``p=1, 3`` will produce a block with
        properties ``a, p = (0, 1), (0, 2), (2, 1), (2, 3)``.

        If ``keys_to_move`` is a set of :py:class:`Labels` and it is empty
        (``len(keys_to_move) == 0``), the :py:class:`Labels.names` will be used
        as if they where passed directly.

        Finally, if ``keys_to_move`` is a non empty set of :py:class:`Labels`,
        the new properties labels will contains **all** of the entries of
        ``keys_to_move`` (regardless of the values taken by
        ``keys_to_move.names`` in the merged blocks' keys) followed by the
        existing properties labels. For example, using ``a=2, 3`` in
        ``keys_to_move``, blocks with properties ``p=1, 2`` will result in
        ``a, p = (2, 1), (2, 2), (3, 1), (3, 2)``. If there is no values (no
        block/missing sample) for a given property in the merged block, then the
        value will be set to zero.

        When using a non empty :py:class:`Labels` for ``keys_to_move``, the
        properties labels of all the merged blocks must take the same values.

        The order of the samples in the merged blocks is controlled by
        ``sort_samples``. If ``sort_samples`` is :py:obj:`True`, samples are
        re-ordered to keep them lexicographically sorted. Otherwise they are
        kept in the order in which they appear in the blocks.

        :param keys_to_move: description of the keys to move
        :param sort_samples: whether to sort the merged samples or keep them in
            the order in which they appear in the original blocks
        :return: a new :py:class:`TensorMap` with merged blocks
        """

    def components_to_properties(self, dimensions: StrSequence) -> "TensorMap":
        """
        Move the given ``dimensions`` from the component labels to the property
        labels for each block.

        :param dimensions: name of the component dimensions to move to the
            properties
        """

    def blocks_matching(self, selection: Labels) -> List[int]:
        """
        Get a (possibly empty) list of block indexes matching the ``selection``.

        This function finds all keys in this :py:class:`TensorMap` with the same
        values as ``selection`` for the dimensions/names contained in the
        ``selection``; and return the corresponding indexes.

        The ``selection`` should contain a single entry.
        """

    def block_by_id(self, index: int) -> TensorBlock:
        """
        Get the block at ``index`` in this :py:class:`TensorMap`.

        :param index: index of the block to retrieve
        """

    def blocks_by_id(self, indices: List[int]) -> List[TensorBlock]:
        """
        Get the blocks with the given ``indices`` in this :py:class:`TensorMap`.

        :param indices: indices of the block to retrieve
        """

    def block(
        self,
        selection: Union[None, int, Labels, LabelsEntry, Dict[str, int]] = None,
    ) -> TensorBlock:
        """
        Get the single block in this :py:class:`TensorMap` matching the ``selection``.

        When ``selection`` is an ``int``, this is equivalent to
        :py:func:`TensorMap.block_by_id`.

        When ``selection`` is an :py:class:`Labels`, it should only contain a single
        entry, which will be used for the selection.

        When ``selection`` is a ``Dict[str, int]``, it is converted into a single single
        :py:class:`LabelsEntry` (the dict keys becoming the names and the dict values
        being joined together to form the :py:class:`LabelsEntry` values), which is then
        used for the selection.

        When ``selection`` is a :py:class:`LabelsEntry`, this function finds the key in
        this :py:class:`TensorMap` with the same values as ``selection`` for the
        dimensions/names contained in the ``selection``; and return the corresponding
        indexes.

        :param selection: description of the block to extract
        """

    def blocks(
        self,
        selection: Union[
            None, List[int], int, Labels, LabelsEntry, Dict[str, int]
        ] = None,
    ) -> List[TensorBlock]:
        """
        Get the blocks in this :py:class:`TensorMap` matching the ``selection``.

        When ``selection`` is ``None`` (the default), all blocks are returned.

        When ``selection`` is an ``int``, this is equivalent to
        :py:func:`TensorMap.block_by_id`; and for a ``List[int]`` this is equivalent to
        :py:func:`TensorMap.blocks_by_id`.

        When ``selection`` is an :py:class:`Labels`, it should only contain a single
        entry, which will be used for the selection.

        When ``selection`` is a ``Dict[str, int]``, it is converted into a single single
        :py:class:`LabelsEntry` (the dict keys becoming the names and the dict values
        being joined together to form the :py:class:`LabelsEntry` values), which is then
        used for the selection.

        When ``selection`` is a :py:class:`LabelsEntry`, this function finds all keys in
        this :py:class:`TensorMap` with the same values as ``selection`` for the
        dimensions/names contained in the ``selection``; and return the corresponding
        blocks.

        :param selection: description of the blocks to extract
        """

    @property
    def sample_names(self) -> List[str]:
        """names of the sample labels for all blocks in this tensor map"""

    @property
    def components_names(self) -> List[List[str]]:
        """names of the component labels for all blocks in this tensor map"""

    @property
    def property_names(self) -> List[str]:
        """names of the property labels for all blocks in this tensor map"""

    def print(self, max_keys: int) -> str:
        """
        Print this :py:class:`TensorMap` to a string, including at most
        ``max_keys`` in the output.

        :param max_keys: how many keys to include in the output. Use ``-1`` to
            include all keys.
        """


def load(path: str) -> TensorMap:
    """
    Load a previously saved :py:class:`TensorMap` from the given path.

    :py:class:`TensorMap` are serialized using numpy's ``.npz`` format, i.e. a
    ZIP file without compression (storage method is ``STORED``), where each file
    is stored as a ``.npy`` array. See the C API documentation for more
    information on the format.

    :param path: path of the file to load
    """


def save(path: str, tensor: TensorMap):
    """
    Save the given :py:class:`TensorMap` to a file at ``path``.

    :py:class:`TensorMap` are serialized using numpy's ``.npz`` format, i.e. a
    ZIP file without compression (storage method is ``STORED``), where each file
    is stored as a ``.npy`` array. See the C API documentation for more
    information on the format.

    :param path: path of the file where to save the data
    :param tensor: tensor to save
    """
