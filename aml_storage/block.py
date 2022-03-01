from ._c_lib import _get_library
from ._c_api import aml_indexes_kind, aml_indexes_t, aml_data_storage_t

import ctypes

from .status import _check_pointer
from .indexes import Indexes

from .data import AmlData, aml_data_to_array


class Block:
    def __init__(self, data, samples: Indexes, symmetric: Indexes, features: Indexes):
        self._lib = _get_library()

        # keep a reference to the data in the block to prevent GC
        self._data = AmlData(data)
        self._gradients = []

        self._ptr = self._lib.aml_block(
            self._data._storage,
            samples._as_aml_indexes_t(),
            symmetric._as_aml_indexes_t(),
            features._as_aml_indexes_t(),
        )
        _check_pointer(self._ptr)
        self._owning = True

    @staticmethod
    def _from_non_owning_ptr(ptr):
        _check_pointer(ptr)
        obj = Block.__new__(Block)
        obj._lib = _get_library()
        obj._ptr = ptr
        obj._owning = False
        obj._data = None
        obj._gradients = []
        return obj

    def __del__(self):
        if hasattr(self, "_lib") and hasattr(self, "_ptr"):
            if self._owning:
                self._lib.aml_block_free(self._ptr)

    def _indexes(self, name, kind):
        result = aml_indexes_t()

        self._lib.aml_block_indexes(self._ptr, name.encode("utf8"), kind.value, result)

        # TODO: keep a reference to the `block` in the Indexes array to ensure
        # it is not removed by GC
        return Indexes._from_aml_indexes_t(result)

    @property
    def samples(self):
        return self._indexes("values", aml_indexes_kind.AML_INDEXES_SAMPLES)

    @property
    def symmetric(self):
        return self._indexes("values", aml_indexes_kind.AML_INDEXES_SYMMETRIC)

    @property
    def features(self):
        return self._indexes("values", aml_indexes_kind.AML_INDEXES_FEATURES)

    @property
    def values(self):
        return self._get_array("values")

    def gradient(self, name):
        assert name != "values"
        data = self._get_array(name)
        samples = self._indexes(name, aml_indexes_kind.AML_INDEXES_SAMPLES)
        return samples, data

    def add_gradient(self, name, samples, gradient):
        gradient = AmlData(gradient)
        self._gradients.append(gradient)

        self._lib.aml_block_add_gradient(
            self._ptr,
            name.encode("utf8"),
            samples._as_aml_indexes_t(),
            gradient._storage,
        )

    def _get_array(self, name):
        data = ctypes.POINTER(aml_data_storage_t)()
        self._lib.aml_block_data(self._ptr, name.encode("utf8"), data)
        return aml_data_to_array(data)
