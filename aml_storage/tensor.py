from typing import List, Union

import ctypes
import numpy as np

from ._c_lib import _get_library
from ._c_api import aml_block_t, aml_labels_t

from .status import _check_pointer
from .labels import Labels, _is_namedtuple
from .block import Block


class TensorMap:
    """
    A TensorMap is the main user-facing class of this library, and can store
    any kind of data used in atomistic machine learning.

    A tensor map contains a list of :py:class:`Block`, each one associated with
    a key. Users can access the blocks either one by one with the
    :py:func:`TensorMap.block` function, or by iterating over the tensor map
    itself:

    .. code-block:: python

        for key, block in tensor:
            ...

    A tensor map provides functions to move some of these keys to the
    samples or properties labels of the blocks, moving from a sparse
    representation of the data to a dense one.
    """

    def __init__(self, keys: Labels, blocks: List[Block]):
        """
        :param keys: keys associated with each block
        :param blocks: set of blocks containing the actual data
        """
        self._lib = _get_library()

        blocks_array_t = ctypes.POINTER(aml_block_t) * len(blocks)
        blocks_array = blocks_array_t(*[block._ptr for block in blocks])

        # all blocks are moved into the tensor map, assign NULL to `block._ptr`
        # to prevent accessing the blocks from Python/double free
        for block in blocks:
            if not block._owning:
                raise ValueError(
                    "can not use blocks from another tensor map in a new one, "
                    "use Block.copy() to make a copy of each block first"
                )

            block._ptr = ctypes.POINTER(aml_block_t)()

        # keep a reference to the blocks in the tensor map in case they contain
        # a Python-allocated array that we need to keep alive
        self._blocks = blocks

        self._ptr = self._lib.aml_tensormap(
            keys._as_aml_labels_t(), blocks_array, len(blocks)
        )

        _check_pointer(self._ptr)

        first_block = self.block(0)
        self.sample_names: List[str] = first_block.samples.names
        """Names of the sample labels for all blocks in this tensor map"""

        self.component_names: List[List[str]] = [
            c.names for c in first_block.components
        ]
        """Names of the component labels for all blocks in this tensor map"""

        self.property_names: List[str] = first_block.properties.names
        """Names of the property labels for all blocks in this tensor map"""

    def __del__(self):
        if hasattr(self, "_lib") and hasattr(self, "_ptr"):
            self._lib.aml_tensormap_free(self._ptr)

    def __iter__(self):
        keys = self.keys
        for i, keys in enumerate(keys):
            yield keys, self._get_block_by_id(i)

    @property
    def keys(self) -> Labels:
        """
        The set of keys labeling the blocks in this tensor map
        """
        result = aml_labels_t()
        self._lib.aml_tensormap_keys(self._ptr, result)
        return Labels._from_aml_labels_t(result, parent=self)

    def block(self, *args, **kwargs) -> Block:
        """
        Get a single block in this tensor map, matching the selection made with
        positional and keyword arguments.

        There are a couple of different ways to call this function:

        .. code-block:: python

            # with a numeric index, this gives a block by its position
            block = tensor.block(3)
            # this block corresponds to tensor.keys[3]

            # with a key
            block = tensor.block(tensor.keys[3])

            # with keyword arguments selecting the block
            block = tensor.block(key=-3, symmetric=4)
            # this assumes `tensor.keys.names == ("key", "symmetric")`

            # with Labels containing a single entry
            labels = Labels(
                names=["key", "symmetric"],
                values=np.array([[-3, 4]], dtype=np.int32)
            )
            block = tensor.block(labels)
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
                    f"got unexpected object in `TensorMap.block`: {type(arg)}"
                )

        selection = Labels(
            kwargs.keys(),
            np.array(list(kwargs.values()), dtype=np.int32).reshape(1, -1),
        )

        return self._block_selection(selection)

    def _get_block_by_id(self, id) -> Block:
        block = ctypes.POINTER(aml_block_t)()
        self._lib.aml_tensormap_block_by_id(self._ptr, block, id)
        return Block._from_ptr(block, parent=self, owning=False)

    def _block_selection(self, selection: Labels) -> Block:
        block = ctypes.POINTER(aml_block_t)()
        self._lib.aml_tensormap_block_selection(
            self._ptr,
            block,
            selection._as_aml_labels_t(),
        )
        return Block._from_ptr(block, parent=self, owning=False)

    def keys_to_properties(self, variables: Union[str, List[str]]):
        """
        Move the given ``variables`` from the keys to the property labels of the
        blocks.

        Blocks containing the same values in the keys for the ``variables`` will
        be merged together. The resulting merged blocks will have ``variables``
        as the first property variables, followed by the current properties. The
        new sample labels will contains all of the merged blocks sample labels,
        re-ordered to keep them lexicographically sorted.

        :param variables: name of the variables to move to the properties
        """
        c_variables = _list_or_str_to_array_c_char(variables)
        self._lib.aml_tensormap_keys_to_properties(
            self._ptr, c_variables, c_variables._length_
        )

    def keys_to_samples(self, variables: Union[str, List[str]]):
        """
        Move the given ``variables`` from the keys to the sample labels of the
        blocks.

        Blocks containing the same values in the keys for the ``variables`` will
        be merged together. The resulting merged blocks will have ``variables``
        as the last sample variables, preceded by the current samples.

        This function is only implemented if all blocks to merge have the same
        property labels.

        :param variables: name of the variables to move to the samples
        """
        c_variables = _list_or_str_to_array_c_char(variables)

        self._lib.aml_tensormap_keys_to_samples(
            self._ptr, c_variables, c_variables._length_
        )

    def components_to_properties(self, variables: Union[str, List[str]]):
        """
        Move the given variables from the component labels to the property labels
        for each block.

        :param variables: name of the component variables to move to the properties
        """
        c_variables = _list_or_str_to_array_c_char(variables)

        self._lib.aml_tensormap_components_to_properties(
            self._ptr, c_variables, c_variables._length_
        )


def _list_or_str_to_array_c_char(strings: Union[str, List[str]]):
    if isinstance(strings, str):
        strings = [strings]

    c_strings = ctypes.ARRAY(ctypes.c_char_p, len(strings))()
    for i, v in enumerate(strings):
        c_strings[i] = v.encode("utf8")

    return c_strings
