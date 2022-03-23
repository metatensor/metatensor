from typing import List, Union

import ctypes
import numpy as np

from ._c_lib import _get_library
from ._c_api import aml_block_t, aml_labels_t

from .status import _check_pointer
from .labels import Labels, _is_namedtuple
from .block import Block


class Descriptor:
    """
    A descriptor is the main user-facing class of this library, and can store
    any kind of data used in atomistic machine learning.

    A descriptor contains a list of :py:class:`Block`, each one associated with
    a label --- called sparse labels. Users can access the blocks either one by
    one with the :py:func:`Descriptor.block` function, or by iterating over the
    descriptor itself:

    .. code-block:: python

        for label, block in descriptor:
            ...

    A descriptor provides functions to move some of these sparse labels to the
    samples or features labels of the blocks, moving from a sparse
    representation of the data to a dense one.
    """

    def __init__(self, sparse: Labels, blocks: List[Block]):
        """
        :param sparse: sparse labels associated with each block
        :param blocks: set of blocks containing the actual data
        """
        self._lib = _get_library()

        blocks_array_t = ctypes.POINTER(aml_block_t) * len(blocks)
        blocks_array = blocks_array_t(*[block._ptr for block in blocks])

        # all blocks are moved into the descriptor, assign NULL to `block._ptr`
        # to prevent accessing the blocks from Python/double free
        for block in blocks:
            block._ptr = ctypes.POINTER(aml_block_t)()

        # keep a reference to the blocks in the descriptor in case they contain
        # a Python-allocated array that we need to keep alive
        self._blocks = blocks

        self._ptr = self._lib.aml_descriptor(
            sparse._as_aml_labels_t(), blocks_array, len(blocks)
        )

        _check_pointer(self._ptr)

        first_block = self.block(0)
        self.sample_names: List[str] = first_block.samples.names
        """Names of the sample labels for all blocks in this descriptor"""

        self.component_names: List[str] = first_block.components.names
        """Names of the component labels for all blocks in this descriptor"""

        self.feature_names: List[str] = first_block.features.names
        """Names of the feature labels for all blocks in this descriptor"""

    def __del__(self):
        if hasattr(self, "_lib") and hasattr(self, "_ptr"):
            self._lib.aml_descriptor_free(self._ptr)

    def __iter__(self):
        sparse = self.sparse
        for i, sparse in enumerate(sparse):
            yield sparse, self._get_block_by_id(i)

    @property
    def sparse(self) -> Labels:
        """
        The set of sparse :py:class:`Labels` labeling the blocks in this
        descriptor
        """
        result = aml_labels_t()
        self._lib.aml_descriptor_sparse_labels(self._ptr, result)
        return Labels._from_aml_labels_t(result, parent=self)

    def block(self, *args, **kwargs) -> Block:
        """
        Get a single block in this descriptor, matching the selection made with
        positional and keyword arguments.

        There are a couple of different ways to call this function:

        .. code-block:: python

            # with a numeric index, this gives a block by its position
            block = descriptor.block(3)
            # this block corresponds to descriptor.sparse[3]

            # with a sparse index entry
            block = descriptor.block(descriptor.sparse[3])

            # with keyword arguments selecting the block
            block = descriptor.block(sparse=-3, symmetric=4)
            # this assumes `descriptor.sparse.names == ("sparse", "symmetric")`

            # with Labels containing a single entry
            labels = Labels(
                names=["sparse", "symmetric"],
                values=np.array([[-3, 4]], dtype=np.int32)
            )
            block = descriptor.block(labels)
        """

        if args:
            if len(args) > 1:
                raise ValueError("only one non-keyword argument is supported")

            arg = args[0]
            if isinstance(arg, int):
                return self._get_block_by_id(arg)
            elif isinstance(arg, Labels):
                return self._block_selection(arg)
            elif isinstance(arg, np.void):
                # single entry from an Labels array
                return self.block(
                    **{name: arg[i] for i, name in enumerate(arg.dtype.names)}
                )
            elif _is_namedtuple(arg):
                return self.block(**arg.as_dict())
            else:
                raise ValueError(
                    f"got unexpected object in `Descriptor.block`: {type(arg)}"
                )

        selection = Labels(
            kwargs.keys(),
            np.array(list(kwargs.values()), dtype=np.int32).reshape(1, -1),
        )

        return self._block_selection(selection)

    def _get_block_by_id(self, id) -> Block:
        block = ctypes.POINTER(aml_block_t)()
        self._lib.aml_descriptor_block_by_id(self._ptr, block, id)
        return Block._from_non_owning_ptr(block, parent=self)

    def _block_selection(self, selection: Labels) -> Block:
        block = ctypes.POINTER(aml_block_t)()
        self._lib.aml_descriptor_block_selection(
            self._ptr,
            block,
            selection._as_aml_labels_t(),
        )
        return Block._from_non_owning_ptr(block, parent=self)

    def sparse_to_features(self, variables: Union[str, List[str]]):
        """
        Move the given variables from the sparse labels to the feature labels of
        the blocks.

        The current blocks will be merged together according to the sparse
        labels remaining after removing ``variables``. The resulting merged
        blocks will have ``variables`` as the first feature variables, followed
        by the current features. The new sample labels will contains all of the
        merged blocks sample labels, re-ordered to keep them lexicographically
        sorted.

        :param variables: name of the sparse variables to move to the features
        """
        c_variables = _list_or_str_to_array_c_char(variables)
        self._lib.aml_descriptor_sparse_to_features(
            self._ptr, c_variables, c_variables._length_
        )

    def sparse_to_samples(self, variables: Union[str, List[str]]):
        """
        Move the given variables from the sparse labels to the sample labels of
        the blocks.

        The current blocks will be merged together according to the sparse
        labels remaining after removing ``variables``. The resulting merged
        blocks will have ``variables`` as the last sample variables, preceded by
        the current samples.

        Currently, this function only works if all merged block have the same
        feature labels.

        :param variables: name of the sparse variables to move to the samples
        """
        c_variables = _list_or_str_to_array_c_char(variables)

        self._lib.aml_descriptor_sparse_to_samples(
            self._ptr, c_variables, c_variables._length_
        )

    def components_to_features(self):
        """
        Move all component labels in each block to the feature labels and
        reshape the data accordingly.
        """
        self._lib.aml_descriptor_components_to_features(self._ptr)


def _list_or_str_to_array_c_char(strings: Union[str, List[str]]):
    if isinstance(strings, str):
        strings = [strings]

    c_strings = ctypes.ARRAY(ctypes.c_char_p, len(strings))()
    for i, v in enumerate(strings):
        c_strings[i] = v.encode("utf8")

    return c_strings
