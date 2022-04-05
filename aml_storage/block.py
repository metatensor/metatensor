import gc
import copy
import ctypes
from typing import Tuple, List

from ._c_lib import _get_library
from ._c_api import aml_label_kind, aml_labels_t, aml_array_t

from .status import _check_pointer
from .labels import Labels

from .data import AmlData, Array, aml_array_t_to_python_object


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
    def _from_ptr(ptr, parent, owning):
        """
        create a block from a pointer, either owning its data (new block as a
        copy of an existing one) or not (block inside a Descriptor)
        """
        _check_pointer(ptr)
        obj = Block.__new__(Block)
        obj._lib = _get_library()
        obj._ptr = ptr
        obj._owning = owning
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

    def __deepcopy__(self, _memodict={}):
        # Temporarily disable garbage collection to ensure the temporary AmlData
        # instance created in _aml_storage_copy will still be available below
        if gc.isenabled():
            gc.disable()
            reset_gc = True
        else:
            reset_gc = False

        new_ptr = self._lib.aml_block_copy(self._ptr)
        copy = Block._from_ptr(new_ptr, parent=None, owning=True)

        # Keep references to the arrays in this block if the arrays were
        # allocated by Python
        try:
            copy._values = aml_array_t_to_python_object(copy._get_raw_array("values"))
        except ValueError:
            # the array was not allocated by Python
            copy._values = None

        for parameter in self.gradients_list():
            try:
                copy._gradients.append(
                    aml_array_t_to_python_object(copy._get_raw_array(parameter))
                )
            except ValueError:
                pass

        if reset_gc:
            gc.enable()

        return copy

    def copy(self) -> "Block":
        """Get a deep copy of this block"""
        return copy.deepcopy(self)

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

    def gradients_list(self) -> List[str]:
        """ """
        parameters = ctypes.POINTER(ctypes.c_char_p)()
        count = ctypes.c_uint64()
        self._lib.aml_block_gradients_list(self._ptr, parameters, count)

        result = []
        for i in range(count.value):
            result.append(parameters[i].decode("utf8"))

        return result

    def has_gradient(self, parameter: str) -> bool:
        """Check if this block contains gradient information with respect to the
        given ``parameter``.

        :param parameter: check for gradients with respect to this ``parameter``
            (e.g. ``positions``, ``cell``, ...)
        """
        return parameter in self.gradients_list()

    def _get_array(self, name: str) -> Array:
        return aml_array_t_to_python_object(self._get_raw_array(name)).array

    def _get_raw_array(self, name: str) -> aml_array_t:
        data = aml_array_t()
        self._lib.aml_block_data(self._ptr, name.encode("utf8"), data)
        return data
