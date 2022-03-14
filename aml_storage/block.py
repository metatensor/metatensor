import ctypes

from ._c_lib import _get_library
from ._c_api import aml_label_kind, aml_labels_t, aml_array_t

from .status import _check_pointer
from .labels import Labels

from .data import AmlData, aml_data_to_array


class Block:
    def __init__(self, data, samples: Labels, components: Labels, features: Labels):
        self._lib = _get_library()

        # keep a reference to the data in the block to prevent GC
        self._data = AmlData(data)
        self._gradients = []

        self._ptr = self._lib.aml_block(
            self._data._storage,
            samples._as_aml_labels_t(),
            components._as_aml_labels_t(),
            features._as_aml_labels_t(),
        )
        self._owning = True
        _check_pointer(self._ptr)

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
        if hasattr(self, "_lib") and hasattr(self, "_ptr") and hasattr(self, "_owning"):
            if self._owning:
                self._lib.aml_block_free(self._ptr)

    def _labels(self, name, kind):
        result = aml_labels_t()

        self._lib.aml_block_labels(self._ptr, name.encode("utf8"), kind.value, result)

        # TODO: keep a reference to the `block` in the Labels array to ensure
        # it is not removed by GC
        return Labels._from_aml_labels_t(result)

    @property
    def samples(self):
        return self._labels("values", aml_label_kind.AML_SAMPLE_LABELS)

    @property
    def components(self):
        return self._labels("values", aml_label_kind.AML_COMPONENTS_LABELS)

    @property
    def features(self):
        return self._labels("values", aml_label_kind.AML_FEATURE_LABELS)

    @property
    def values(self):
        return self._get_array("values")

    def gradient(self, name):
        assert name != "values"
        data = self._get_array(name)
        samples = self._labels(name, aml_label_kind.AML_SAMPLE_LABELS)
        return samples, data

    def add_gradient(self, name, samples, gradient):
        gradient = AmlData(gradient)
        self._gradients.append(gradient)

        self._lib.aml_block_add_gradient(
            self._ptr,
            name.encode("utf8"),
            samples._as_aml_labels_t(),
            gradient._storage,
        )

    def has_gradient(self, name):
        result = ctypes.c_bool()
        self._lib.aml_block_has_gradient(self._ptr, name.encode("utf8"), result)
        return result.value

    def _get_array(self, name):
        data = ctypes.POINTER(aml_array_t)()
        self._lib.aml_block_data(self._ptr, name.encode("utf8"), data)
        return aml_data_to_array(data)
