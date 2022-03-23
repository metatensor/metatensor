import ctypes
from typing import Tuple

from ._c_lib import _get_library
from ._c_api import aml_label_kind, aml_labels_t, aml_array_t

from .status import _check_pointer
from .labels import Labels

from .data import AmlData, aml_data_to_array, Array


class Block:
    """
    Basic building block for descriptor. A single block contains a 3-dimensional
    :py:class:`aml_storage.data.Array`, and three sets of :py:class:`Labels`
    (one for each dimension).

    A block can also contain gradients of the values with respect to a variety
    of parameters. In this case, each gradient has a separate set of samples,
    but share the same components and feature labels as the values.
    """

    def __init__(
        self,
        values: Array,
        samples: Labels,
        components: Labels,
        features: Labels,
    ):
        """
        :param values: array containing the data for this block
        :param samples: labels describing the samples (first dimension of the
            array)
        :param components: labels describing the components (second dimension of
            the array). This is set to :py:func:`Labels.single` when dealing
            with scalar/invariant values.
        :param features: labels describing the samples (third dimension of the
            array)
        """
        self._lib = _get_library()

        # keep a reference to the values in the block to prevent GC
        self._values = AmlData(values)
        self._gradients = []

        self._ptr = self._lib.aml_block(
            self._values._storage,
            samples._as_aml_labels_t(),
            components._as_aml_labels_t(),
            features._as_aml_labels_t(),
        )
        self._owning = True
        self._parent = None
        _check_pointer(self._ptr)

    @staticmethod
    def _from_non_owning_ptr(ptr, parent):
        """create a block not owning its data (i.e. a block inside a Descriptor)"""
        _check_pointer(ptr)
        obj = Block.__new__(Block)
        obj._lib = _get_library()
        obj._ptr = ptr
        obj._owning = False
        obj._values = None
        obj._gradients = []
        # keep a reference to the parent object (usually a Descriptor) to
        # prevent it from beeing garbage-collected & removing this block
        obj._parent = parent
        return obj

    def __del__(self):
        if hasattr(self, "_lib") and hasattr(self, "_ptr") and hasattr(self, "_owning"):
            if self._owning:
                self._lib.aml_block_free(self._ptr)

    def _labels(self, name: str, kind) -> Labels:
        """Get the labels of the given ``kind`` for the ``name`` array in this
        block"""
        result = aml_labels_t()
        self._lib.aml_block_labels(self._ptr, name.encode("utf8"), kind.value, result)
        return Labels._from_aml_labels_t(result, parent=self)

    @property
    def samples(self) -> Labels:
        """
        Access the sample :py:class:`Labels` for this block. The entries in
        these labels describe the first dimension of the ``values`` array.
        """
        return self._labels("values", aml_label_kind.AML_SAMPLE_LABELS)

    @property
    def components(self) -> Labels:
        """
        Access the component :py:class:`Labels` for this block. The entries in
        these labels describe the second dimension of the ``values`` array, and
        any additional gradient stored in this block.
        """
        return self._labels("values", aml_label_kind.AML_COMPONENTS_LABELS)

    @property
    def features(self) -> Labels:
        """
        Access the feature :py:class:`Labels` for this block. The entries in
        these labels describe the third dimension of the ``values`` array, and
        any additional gradient stored in this block.
        """
        return self._labels("values", aml_label_kind.AML_FEATURE_LABELS)

    @property
    def values(self) -> Array:
        """
        Access the values for this block. Values are stored as a 3-dimensional
        array of shape ``(samples, components, features)``.

        The array type depends on how the block was created. Currently, numpy
        ``ndarray`` and torch ``Tensor`` are supported.
        """
        return self._get_array("values")

    def gradient(self, parameter: str) -> Tuple[Labels, Array]:
        """
        Get the gradient of the ``values`` in this block with respect to
        ``parameter``, as well as the corresponding gradient samples.
        """
        assert parameter != "values"
        data = self._get_array(parameter)
        samples = self._labels(parameter, aml_label_kind.AML_SAMPLE_LABELS)
        return samples, data

    def add_gradient(self, parameter: str, gradient_samples: Labels, gradient: Array):
        """Add a set of gradients with respect to ``parameters`` in this block.

        :param parameter: add gradients with respect to this ``parameter`` (e.g.
            ``positions``, ``cell``, ...)
        :param gradient_samples: labels describing the gradient samples
        :param gradient: the gradient array, of shape ``(gradient_samples,
            components, features)``, where the components and features labels
            are the same as the values components and features labels.
        """
        gradient = AmlData(gradient)
        self._gradients.append(gradient)

        self._lib.aml_block_add_gradient(
            self._ptr,
            parameter.encode("utf8"),
            gradient_samples._as_aml_labels_t(),
            gradient._storage,
        )

    def has_gradient(self, parameter: str) -> bool:
        """Check if this block contains gradient information with respect to the
        given ``parameter``.

        :param parameter: check for gradients with respect to this ``parameter``
            (e.g. ``positions``, ``cell``, ...)
        """
        result = ctypes.c_bool()
        self._lib.aml_block_has_gradient(self._ptr, parameter.encode("utf8"), result)
        return result.value

    def _get_array(self, name: str) -> Array:
        data = ctypes.POINTER(aml_array_t)()
        self._lib.aml_block_data(self._ptr, name.encode("utf8"), data)
        return aml_data_to_array(data)
