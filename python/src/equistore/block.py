import copy
import ctypes
from typing import Generator, List, Tuple

from ._c_api import c_uintptr_t, eqs_array_t, eqs_labels_t
from ._c_lib import _get_library
from .data import Array, ArrayWrapper, eqs_array_to_python_array
from .labels import Labels
from .status import _check_pointer


class TensorBlock:
    """
    Basic building block for a tensor map.

    A single block contains a n-dimensional :py:class:`equistore.data.Array`,
    and n sets of :py:class:`Labels` (one for each dimension). The first
    dimension is the *samples* dimension, the last dimension is the *properties*
    dimension. Any intermediate dimension is called a *component* dimension.

    Samples should be used to describe *what* we are representing, while
    properties should contain information about *how* we are representing it.
    Finally, components should be used to describe vectorial or tensorial
    components of the data.

    A block can also contain gradients of the values with respect to a variety
    of parameters. In this case, each gradient has a separate set of samples,
    and possibly components but share the same property labels as the values.
    """

    def __init__(
        self,
        values: Array,
        samples: Labels,
        components: List[Labels],
        properties: Labels,
    ):
        """
        :param values: array containing the data for this block
        :param samples: labels describing the samples (first dimension of the
            array)
        :param components: labels describing the components (second dimension of
            the array). This is set to :py:func:`Labels.single` when dealing
            with scalar/invariant values.
        :param properties: labels describing the samples (third dimension of the
            array)
        """
        self._lib = _get_library()
        self._parent = None

        components_array = ctypes.ARRAY(eqs_labels_t, len(components))()
        for i, component in enumerate(components):
            components_array[i] = component._as_eqs_labels_t()

        values = ArrayWrapper(values)

        self._actual_ptr = self._lib.eqs_block(
            values.into_eqs_array(),
            samples._as_eqs_labels_t(),
            components_array,
            len(components_array),
            properties._as_eqs_labels_t(),
        )
        _check_pointer(self._actual_ptr)

    @staticmethod
    def _from_ptr(ptr, parent):
        """
        create a block from a pointer, either owning its data (new block as a
        copy of an existing one) or not (block inside a :py:class:`TensorMap`)
        """
        _check_pointer(ptr)
        obj = TensorBlock.__new__(TensorBlock)
        obj._lib = _get_library()
        obj._actual_ptr = ptr
        # keep a reference to the parent object (usually a TensorMap) to
        # prevent it from beeing garbage-collected & removing this block
        obj._parent = parent
        return obj

    @property
    def _ptr(self):
        if self._actual_ptr is None:
            raise ValueError(
                "this block has been moved inside a TensorMap and can no longer be used"
            )

        return self._actual_ptr

    def _move_ptr(self):
        assert self._parent is None
        self._actual_ptr = None

    def __del__(self):
        if (
            hasattr(self, "_lib")
            and self._lib is not None
            and hasattr(self, "_actual_ptr")
            and hasattr(self, "_parent")
        ):
            if self._parent is None:
                self._lib.eqs_block_free(self._actual_ptr)

    def __deepcopy__(self, _memodict):
        new_ptr = self._lib.eqs_block_copy(self._ptr)
        return TensorBlock._from_ptr(new_ptr, parent=None)

    def copy(self) -> "TensorBlock":
        """
        Get a deep copy of this block, including all the (potentially non
        Python-owned) data and metadata
        """
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        s = "TensorBlock\n"
        s += f"    samples ({len(self.samples)}): {str(list(self.samples.names))}"
        s += "\n"
        s += "    components ("
        s += ", ".join([str(len(c)) for c in self.components])
        s += "): ["
        for ic in self.components:
            for name in ic.names[:]:
                s += "'" + name + "', "
        if len(self.components) > 0:
            s = s[:-2]
        s += "]\n"
        s += f"    properties ({len(self.properties)}): "
        s += f"{str(list(self.properties.names))}\n"
        s += "    gradients: "
        if len(self.gradients_list()) > 0:
            s += f"{str(list(self.gradients_list()))}"
        else:
            s += "no"

        return s

    @property
    def values(self) -> Array:
        """
        Access the values for this block.

        The array type depends on how the block was created. Currently, numpy
        ``ndarray`` and torch ``Tensor`` are supported.
        """

        raw_array = _get_raw_array(self._lib, self._ptr, "values")
        return eqs_array_to_python_array(raw_array, parent=self)

    @property
    def samples(self) -> Labels:
        """
        Access the sample :py:class:`Labels` for this block.

        The entries in these labels describe the first dimension of the
        ``values`` array.
        """
        return self._labels(0)

    @property
    def components(self) -> List[Labels]:
        """
        Access the component :py:class:`Labels` for this block.

        The entries in these labels describe intermediate dimensions of the
        ``values`` array.
        """
        n_components = len(self.values.shape) - 2

        result = []
        for axis in range(n_components):
            result.append(self._labels(axis + 1))

        return result

    @property
    def properties(self) -> Labels:
        """
        Access the property :py:class:`Labels` for this block.

        The entries in these labels describe the last dimension of the
        ``values`` array. The properties are guaranteed to be the same for
        values and gradients in the same block.
        """
        property_axis = len(self.values.shape) - 1
        return self._labels(property_axis)

    def _labels(self, axis) -> Labels:
        result = eqs_labels_t()
        self._lib.eqs_block_labels(self._ptr, "values".encode("utf8"), axis, result)
        return Labels._from_eqs_labels_t(result)

    def gradient(self, parameter: str) -> "Gradient":
        """
        Get the gradient of the ``values`` in this block with respect to
        the given ``parameter``.

        :param parameter: check for gradients with respect to this ``parameter``
            (e.g. ``positions``, ``cell``, ...)
        """
        if not self.has_gradient(parameter):
            raise ValueError(
                f"this block does not contain gradient with respect to {parameter}"
            )
        return Gradient(self, parameter)

    def add_gradient(
        self,
        parameter: str,
        data: Array,
        samples: Labels,
        components: List[Labels],
    ):
        """
        Add a set of gradients with respect to ``parameters`` in this block.

        :param parameter: add gradients with respect to this ``parameter`` (e.g.
            ``positions``, ``cell``, ...)
        :param data: the gradient array, of shape ``(gradient_samples,
            components, properties)``, where the properties labels are the same
            as the values' properties labels.
        :param samples: labels describing the gradient samples
        :param components: labels describing the gradient components
        """
        if self._parent is not None:
            raise ValueError(
                "can not add gradient on this block since it is a view inside "
                "a TensorMap"
            )
        components_array = ctypes.ARRAY(eqs_labels_t, len(components))()
        for i, component in enumerate(components):
            components_array[i] = component._as_eqs_labels_t()

        data = ArrayWrapper(data)

        self._lib.eqs_block_add_gradient(
            self._ptr,
            parameter.encode("utf8"),
            data.into_eqs_array(),
            samples._as_eqs_labels_t(),
            components_array,
            len(components_array),
        )

    def gradients_list(self) -> List[str]:
        """Get a list of all gradients defined in this block."""
        parameters = ctypes.POINTER(ctypes.c_char_p)()
        count = c_uintptr_t()
        self._lib.eqs_block_gradients_list(self._ptr, parameters, count)

        result = []
        for i in range(count.value):
            result.append(parameters[i].decode("utf8"))

        return result

    def has_gradient(self, parameter: str) -> bool:
        """
        Check if this block contains gradient information with respect to the
        given ``parameter``.

        :param parameter: check for gradients with respect to this ``parameter``
            (e.g. ``positions``, ``cell``, ...)
        """
        return parameter in self.gradients_list()

    def gradients(self) -> Generator[Tuple[str, "Gradient"], None, None]:
        """Get an iterator over all gradients defined in this block."""
        for parameter in self.gradients_list():
            yield (parameter, self.gradient(parameter))


class Gradient:
    """
    Proxy class allowing to access the information associated with a gradient
    inside a block.
    """

    def __init__(self, block: TensorBlock, name: str):
        self._lib = _get_library()

        self._block = block
        self._name = name

    def __repr__(self) -> str:
        s = "Gradient TensorBlock\n"
        s += "parameter: '{}'\n".format(self._name)
        s += f"samples ({len(self.samples)}): {str(list(self.samples.names))}"
        s += "\n"
        s += "components ("
        s += ", ".join([str(len(c)) for c in self.components])
        s += "): ["
        for ic in self.components:
            for name in ic.names[:]:
                s += "'" + name + "', "
        if len(self.components) > 0:
            s = s[:-2]
        s += "]\n"
        s += f"properties ({len(self.properties)}): {str(list(self.properties.names))}"

        return s

    @property
    def data(self) -> Array:
        """
        Access the data for this gradient.

        The array type depends on how the block was created. Currently, numpy
        ``ndarray`` and torch ``Tensor`` are supported.
        """

        raw_array = _get_raw_array(self._lib, self._block._ptr, self._name)
        return eqs_array_to_python_array(raw_array, parent=self)

    @property
    def samples(self) -> Labels:
        """
        Access the sample :py:class:`Labels` for this gradient.

        The entries in these labels describe the first dimension of the ``data``
        array.
        """
        return self._labels(0)

    @property
    def components(self) -> List[Labels]:
        """
        Access the component :py:class:`Labels` for this gradient.

        The entries in these labels describe intermediate dimensions of the
        ``data`` array.
        """
        n_components = len(self.data.shape) - 2

        result = []
        for axis in range(n_components):
            result.append(self._labels(axis + 1))

        return result

    @property
    def properties(self) -> Labels:
        """
        Access the property :py:class:`Labels` for this gradient.

        The entries in these labels describe the last dimension of the ``data``
        array. The properties are guaranteed to be the same for values and
        gradients in the same block.
        """
        property_axis = len(self.data.shape) - 1
        return self._labels(property_axis)

    def _labels(self, axis) -> Labels:
        result = eqs_labels_t()
        self._lib.eqs_block_labels(
            self._block._ptr, self._name.encode("utf8"), axis, result
        )
        return Labels._from_eqs_labels_t(result)


def _get_raw_array(lib, block_ptr, name) -> eqs_array_t:
    data = eqs_array_t()
    lib.eqs_block_data(block_ptr, name.encode("utf8"), data)
    return data
