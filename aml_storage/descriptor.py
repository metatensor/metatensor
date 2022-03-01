from typing import List, Union

import ctypes
import numpy as np

from ._c_lib import _get_library
from ._c_api import aml_block_t, aml_indexes_t

from .status import _check_pointer
from .indexes import Indexes, _is_namedtuple
from .block import Block


class Descriptor:
    def __init__(self, sparse_indexes: Indexes, blocks: List[Block]):
        self._lib = _get_library()

        blocks_array_t = ctypes.POINTER(aml_block_t) * len(blocks)
        blocks_array = blocks_array_t(*[block._ptr for block in blocks])

        # all blocks are moved into the descriptor, assign NULL to `block._ptr`
        for block in blocks:
            block._ptr = ctypes.POINTER(aml_block_t)()

        # keep a reference to the blocks in the descriptor in case they contain
        # a Python-allocated array that we need to keep alive
        self._blocks = blocks

        self._ptr = self._lib.aml_descriptor(
            sparse_indexes._as_aml_indexes_t(), blocks_array, len(blocks)
        )

        _check_pointer(self._ptr)

    def __del__(self):
        if hasattr(self, "_lib") and hasattr(self, "_ptr"):
            self._lib.aml_descriptor_free(self._ptr)

    def __iter__(self):
        sparse = self.sparse_indexes
        for i, sparse in enumerate(sparse):
            yield sparse, self._get_block_by_id(i)

    @property
    def sparse_indexes(self):
        result = aml_indexes_t()

        self._lib.aml_descriptor_sparse_indexes(self._ptr, result)

        # TODO: keep a reference to the `descriptor` in the Indexes array to
        # ensure it is not removed by GC
        return Indexes._from_aml_indexes_t(result)

    def block(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise ValueError("only one non-keyword argument is supported")

            arg = args[0]
            if isinstance(arg, int):
                return self._get_block_by_id(arg)
            elif isinstance(arg, Indexes):
                return self._block_selection(arg)
            elif isinstance(arg, np.void):
                # single entry from an Indexes array
                return self.block(
                    **{name: arg[i] for i, name in enumerate(arg.dtype.names)}
                )
            elif _is_namedtuple(arg):
                return self.block(**arg.as_dict())
            else:
                raise ValueError(
                    f"got unexpected object in `Descriptor.block`: {type(arg)}"
                )

        selection = Indexes(
            kwargs.keys(),
            np.array(list(kwargs.values()), dtype=np.int32).reshape(1, -1),
        )

        return self._block_selection(selection)

    def _get_block_by_id(self, id):
        block = ctypes.POINTER(aml_block_t)()

        self._lib.aml_descriptor_block_by_id(self._ptr, block, id)

        # TODO: keep a reference to the `descriptor` in the block to ensure it
        # is not removed by GC
        return Block._from_non_owning_ptr(block)

    def _block_selection(self, selection: Indexes):
        block = ctypes.POINTER(aml_block_t)()

        self._lib.aml_descriptor_block_selection(
            self._ptr,
            block,
            selection._as_aml_indexes_t(),
        )

        # TODO: keep a reference to the `descriptor` in the block to ensure it
        # is not removed by GC
        return Block._from_non_owning_ptr(block)

    def sparse_to_features(self, variables: Union[str, List[str]]):
        c_variables = _list_str_to_array_c_char(variables)
        self._lib.aml_descriptor_sparse_to_features(
            self._ptr, c_variables, c_variables._length_
        )

    def sparse_to_samples(self, variables: Union[str, List[str]]):
        c_variables = _list_str_to_array_c_char(variables)

        self._lib.aml_descriptor_sparse_to_samples(
            self._ptr, c_variables, c_variables._length_
        )

    def symmetric_to_features(self):
        self._lib.aml_descriptor_symmetric_to_features(self._ptr)


def _list_str_to_array_c_char(strings: Union[str, List[str]]):
    if isinstance(strings, str):
        strings = [strings]

    c_strings = ctypes.ARRAY(ctypes.c_char_p, len(strings))()
    for i, v in enumerate(strings):
        c_strings[i] = v.encode("utf8")

    return c_strings
