import copy
import ctypes
from typing import Generator, List, Tuple

from ._c_api import c_uintptr_t, eqs_array_t, eqs_block_t, eqs_labels_t
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
    of parameters. In this case, each gradient is a :py:class:`TensorBlock` with
    a separate set of samples and possibly components, but which shares the same
    property labels as the original :py:class:`TensorBlock`.
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
                "this block has been moved inside a TensorMap/another TensorBlock "
                "and can no longer be used"
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

    def __copy__(self):
        raise ValueError(
            "shallow copies of TensorBlock are not possible, use a deepcopy instead"
        )

    def __deepcopy__(self, _memodict):
        new_ptr = self._lib.eqs_block_copy(self._ptr)
        return TensorBlock._from_ptr(new_ptr, parent=None)

    def __reduce__(self):
        raise NotImplementedError(
            "Pickling for TensorBlocks does not work. Please wrap it into a TensorMap"
            " to save it."
        )

    def copy(self) -> "TensorBlock":
        """
        Get a deep copy of this block, including all the (potentially non
        Python-owned) data and metadata
        """
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        if isinstance(self._parent, TensorBlock):
            # TODO: print the gradient parameter as well
            s = "Gradient TensorBlock\n"
        else:
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
            s += "None"

        return s

    def __eq__(self, other):
        from equistore.operations import equal_block

        return equal_block(self, other)

    def __ne__(self, other):
        from equistore.operations import equal_block

        return not equal_block(self, other)

    @property
    def values(self) -> Array:
        """
        Access the values for this block.

        The array type depends on how the block was created. Currently, numpy
        ``ndarray`` and torch ``Tensor`` are supported.
        """

        raw_array = _get_raw_array(self._lib, self._ptr)
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
        self._lib.eqs_block_labels(self._ptr, axis, result)
        return Labels._from_eqs_labels_t(result)

    def gradient(self, parameter: str) -> "TensorBlock":
        """
        Get the gradient of the block ``values``  with respect to the given
        ``parameter``.

        :param parameter: check for gradients with respect to this ``parameter``
            (e.g. ``positions``, ``cell``, ...)

        >>> import numpy as np
        >>> from equistore import TensorBlock, Labels
        >>> block = TensorBlock(
        ...     values=np.full((3, 1, 5), 1.0),
        ...     samples=Labels(["structure"], np.array([[0], [2], [4]])),
        ...     components=[Labels.arange("component", 1)],
        ...     properties=Labels.arange("property", 5),
        ... )
        >>> positions_gradient = TensorBlock(
        ...     values=np.full((2, 3, 1, 5), 11.0),
        ...     samples=Labels(["sample", "atom"], np.array([[0, 2], [2, 3]])),
        ...     components=[
        ...         Labels.arange("direction", 3),
        ...         Labels.arange("component", 1),
        ...     ],
        ...     properties=Labels.arange("property", 5),
        ... )

        >>> block.add_gradient("positions", positions_gradient)
        >>> cell_gradient = TensorBlock(
        ...     values=np.full((2, 3, 3, 1, 5), 15.0),
        ...     samples=Labels.arange("sample", 2),
        ...     components=[
        ...         Labels.arange("direction_1", 3),
        ...         Labels.arange("direction_2", 3),
        ...         Labels.arange("component", 1),
        ...     ],
        ...     properties=Labels.arange("property", 5),
        ... )
        >>> block.add_gradient("cell", cell_gradient)

        >>> positions_gradient = block.gradient("positions")
        >>> print(positions_gradient)
        Gradient TensorBlock
            samples (2): ['sample', 'atom']
            components (3, 1): ['direction', 'component']
            properties (5): ['property']
            gradients: None

        >>> cell_gradient = block.gradient("cell")
        >>> print(cell_gradient)
        Gradient TensorBlock
            samples (2): ['sample']
            components (3, 3, 1): ['direction_1', 'direction_2', 'component']
            properties (5): ['property']
            gradients: None
        """
        gradient_block = ctypes.POINTER(eqs_block_t)()
        self._lib.eqs_block_gradient(
            self._ptr, parameter.encode("utf8"), gradient_block
        )

        return TensorBlock._from_ptr(gradient_block, parent=self)

    def add_gradient(self, parameter: str, gradient: "TensorBlock"):
        """
        Add gradient with respect to ``parameter`` in this block.

        :param parameter:
            add gradients with respect to this ``parameter`` (e.g.
            ``positions``, ``cell``, ...)

        :param gradient:
            a :py:class:`TensorBlock` whose values contain the gradients with
            respect to the ``parameter``. The labels of the gradient
            :py:class:`TensorBlock` should be organized as follows: its
            ``samples`` must contain ``"sample"`` as the first label, which
            establishes a correspondence with the ``samples`` of the original
            :py:class:`TensorBlock`; its components must contain at least the
            same components as the original :py:class:`TensorBlock`, with any
            additional :py:class:`Labels` coming before those; its properties
            must match those of the original :py:class:`TensorBlock`.

        >>> import numpy as np
        >>> from equistore import TensorBlock, Labels
        >>> block = TensorBlock(
        ...     values=np.full((3, 1, 1), 1.0),
        ...     samples=Labels(["structure"], np.array([[0], [2], [4]])),
        ...     components=[Labels.arange("component", 1)],
        ...     properties=Labels.arange("property", 1),
        ... )
        >>> gradient = TensorBlock(
        ...     values=np.full((2, 1, 1), 11.0),
        ...     samples=Labels(["sample", "parameter"], np.array([[0, -2], [2, 3]])),
        ...     components=[Labels.arange("component", 1)],
        ...     properties=Labels.arange("property", 1),
        ... )
        >>> block.add_gradient("parameter", gradient)
        >>> print(block)
        TensorBlock
            samples (3): ['structure']
            components (1): ['component']
            properties (1): ['property']
            gradients: ['parameter']
        """
        if self._parent is not None:
            raise ValueError(
                "cannot add gradient on this block since it is a view inside "
                "a TensorMap or another TensorBlock"
            )

        gradient_ptr = gradient._ptr

        # the gradient is moved inside this block, assign NULL to
        # `gradient._ptr` to prevent accessing invalid data from Python and
        # double free
        gradient._move_ptr()

        self._lib.eqs_block_add_gradient(
            self._ptr, parameter.encode("utf8"), gradient_ptr
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

    def gradients(self) -> Generator[Tuple[str, "TensorBlock"], None, None]:
        """Get an iterator over all gradients defined in this block."""
        for parameter in self.gradients_list():
            yield (parameter, self.gradient(parameter))


def _get_raw_array(lib, block_ptr) -> eqs_array_t:
    data = eqs_array_t()
    lib.eqs_block_data(block_ptr, data)
    return data
