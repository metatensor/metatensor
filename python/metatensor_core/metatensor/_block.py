import copy
import ctypes
import pathlib
from pickle import PickleBuffer
from typing import Any, BinaryIO, Generator, List, Sequence, Tuple, Union

from . import _data
from ._c_api import c_uintptr_t, mts_array_t, mts_block_t, mts_labels_t
from ._c_lib import _get_library
from ._data import (
    Array,
    Device,
    DType,
)
from ._labels import Labels
from ._status import check_pointer


class TensorBlock:
    """
    Basic building block for a :py:class:`TensorMap`.

    A single block contains a n-dimensional :py:class:`metatensor.data.Array`,
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

    >>> import numpy as np
    >>> block = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1, 2, 4],
    ...             [3, 5, 6],
    ...         ]
    ...     ),
    ...     samples=Labels("samples", np.array([[4], [2]])),
    ...     components=[],
    ...     properties=Labels("properties", np.array([[0], [1], [2]])),
    ... )
    >>> block
    TensorBlock
        samples (2): ['samples']
        components (): []
        properties (3): ['properties']
        gradients: None
    >>> block.samples
    Labels(
        samples
           4
           2
    )
    >>> block.values[block.samples.position([2])]
    array([3, 5, 6])
    """

    def __init__(
        self,
        values: Array,
        samples: Labels,
        components: Sequence[Labels],
        properties: Labels,
    ):
        """
        :param values: array containing the values for this block
        :param samples: labels describing the samples (first dimension of the array)
        :param components: list of labels describing the components (intermediate
            dimensions of the array). This should be an empty list for scalar/invariant
            data.
        :param properties: labels describing the properties (last dimension of the
            array)
        """
        self._lib = _get_library()
        self._parent = None
        self._gradient_parameters = []

        if not isinstance(samples, Labels):
            raise TypeError(f"`samples` must be metatensor Labels, not {type(samples)}")

        components = list(components)
        for component in components:
            if not isinstance(component, Labels):
                raise TypeError(
                    "`components` elements must be metatensor Labels, "
                    f"not {type(component)}"
                )

        if not isinstance(properties, Labels):
            raise TypeError(
                f"`properties` must be metatensor Labels, not {type(properties)}"
            )

        components_array = ctypes.ARRAY(ctypes.POINTER(mts_labels_t), len(components))()
        for i, component in enumerate(components):
            components_array[i] = component.as_mts_labels_t()

        mts_array = _data.create_mts_array(values)
        self._ptr = self._lib.mts_block(
            mts_array,
            samples.as_mts_labels_t(),
            components_array,
            len(components_array),
            properties.as_mts_labels_t(),
        )
        check_pointer(self._ptr)

        self._cached_dtype = _data.array_dtype(values)
        self._cached_device = _data.array_device(values)

    @staticmethod
    def unsafe_from_ptr(block: ctypes.POINTER(mts_block_t)):
        """
        Create a :py:class:`TensorBlock` from a raw ``mts_block_t`` pointer.

        The :py:class:`TensorBlock` takes ownership of the pointer, and will
        release the corresponding memory when garbage-collected.
        """
        assert block, "mts_block_t pointer is null"
        obj = TensorBlock.__new__(TensorBlock)
        obj._lib = _get_library()
        obj._gradient_parameters = []
        obj._ptr = block
        obj._cached_dtype = None
        obj._cached_device = None
        obj._parent = None
        return obj

    @staticmethod
    def unsafe_view_from_ptr(ptr: ctypes.POINTER(mts_block_t), parent: Any):
        """
        Create a :py:class:`TensorBlock` from a raw ``mts_block_t`` pointer, keeping a
        reference to the ``parent`` to prevent garbage collection.

        The :py:class:`TensorBlock` does **not** take ownership of the pointer, and will
        not release the corresponding memory.
        """
        assert parent is not None, (
            "please use TensorBlock.unsafe_from_ptr to take ownership of a pointer"
        )

        obj = TensorBlock.unsafe_from_ptr(ptr)
        # keep a reference to the parent object (usually a TensorMap) to
        # prevent it from being garbage-collected & removing this block
        obj._parent = parent
        return obj

    def as_mts_block_t(self) -> ctypes.POINTER(mts_block_t):
        """
        Get the underlying C pointer for this :py:class:`TensorBlock`.

        This class still manages the block memory after the call. Use
        :py:meth:`TensorBlock.release` to take ownership of the pointer.
        """
        if not self._ptr:
            raise ValueError(
                "this block has been released or moved inside a TensorBlock "
                "or TensorMap and can no longer be used"
            )

        return self._ptr

    def release(self):
        """
        Release the underlying C pointer of this :py:class:`TensorBlock`.

        This class is no longer managing the block memory after the call, the
        user is expected to re-create a :py:class:`TensorBlock` with
        :py:meth:`TensorBlock.unsafe_from_ptr`, or pass the pointer to a C
        function that will call ``mts_block_free``.
        """
        if self._parent is not None:
            raise RuntimeError(
                "can not release this TensorBlock, it is a view inside another "
                "TensorBlock or a TensorMap"
            )

        ptr = self.as_mts_block_t()
        self._ptr = None
        return ptr

    def __del__(self):
        if (
            hasattr(self, "_lib")
            and self._lib is not None
            and hasattr(self, "_ptr")
            and hasattr(self, "_parent")
        ):
            if self._parent is None:
                self._lib.mts_block_free(self._ptr)

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, _memodict):
        return self.copy(deep=True)

    def __reduce__(self):
        raise NotImplementedError(
            "Pickling for is not implemented for TensorBlocks, wrap the block in a "
            "TensorMap first"
        )

    def __len__(self) -> int:
        """
        Get the length of the values stored in this block
        (i.e. the number of samples in the block)
        """
        return len(self.values)

    @property
    def shape(self):
        """
        Get the shape of the values  array in this block.
        """
        return self.values.shape

    def copy(self, deep: bool = True) -> "TensorBlock":
        """
        Get a copy of this block, with the same values and labels. If ``deep`` is
        ``True``, also make a full copy of the values; otherwise, the values in the new
        block will share the same memory as those in this block.

        :param deep: if ``True``, create a deep copy of the block
        """
        if deep:
            new_ptr = self._lib.mts_block_copy(self.as_mts_block_t())
            check_pointer(new_ptr)
            return TensorBlock.unsafe_from_ptr(new_ptr)
        else:
            new_block = TensorBlock(
                values=self.values,
                samples=self.samples,
                components=self.components,
                properties=self.properties,
            )

            for parameter in self.gradients_list():
                gradient = self.gradient(parameter)
                new_block.add_gradient(parameter, gradient.copy(deep=False))

            return new_block

    def __repr__(self) -> str:
        if not self._ptr:
            # The block has been released
            return "TensorBlock(<empty>)"

        if len(self._gradient_parameters) != 0:
            s = f"Gradient TensorBlock ('{'/'.join(self._gradient_parameters)}')\n"
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
        from metatensor.operations import equal_block

        return equal_block(self, other)

    def __ne__(self, other):
        from metatensor.operations import equal_block

        return not equal_block(self, other)

    @property
    def _raw_values(self) -> mts_array_t:
        """Get the raw ``mts_array_t`` corresponding to this block's values"""
        data = mts_array_t()
        self._lib.mts_block_data(self.as_mts_block_t(), data)
        return data

    @property
    def values(self) -> Array:
        """
        Get the values for this block.

        The array type depends on how the block was created. Currently, numpy
        ``ndarray`` and torch ``Tensor`` are supported.
        """

        return _data.mts_array_to_python_array(self._raw_values, parent=self)

    @values.setter
    def values(self, new_values):
        raise AttributeError(
            "Direct assignment to `values` is not possible. "
            "Please use block.values[:] = new_values instead."
        )

    @property
    def samples(self) -> Labels:
        """
        Get the sample :py:class:`Labels` for this block.

        The entries in these labels describe the first dimension of the
        ``values`` array.
        """
        return self._labels(0)

    @property
    def components(self) -> List[Labels]:
        """
        Get the component :py:class:`Labels` for this block.

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
        Get the property :py:class:`Labels` for this block.

        The entries in these labels describe the last dimension of the
        ``values`` array. The properties are guaranteed to be the same for
        values and gradients in the same block.
        """
        property_axis = len(self.values.shape) - 1
        return self._labels(property_axis)

    def _labels(self, axis) -> Labels:
        result = self._lib.mts_block_labels(self.as_mts_block_t(), axis)
        check_pointer(result)
        return Labels.unsafe_from_ptr(result)

    def gradient(self, parameter: str) -> "TensorBlock":
        """
        Get the gradient of the block ``values``  with respect to the given
        ``parameter``.

        :param parameter: check for gradients with respect to this ``parameter``
            (e.g. ``positions``, ``cell``, ...)

        >>> import numpy as np
        >>> from metatensor import Labels, TensorBlock
        >>> block = TensorBlock(
        ...     values=np.full((3, 1, 5), 1.0),
        ...     samples=Labels(["system"], np.array([[0], [2], [4]])),
        ...     components=[Labels.range("component", 1)],
        ...     properties=Labels.range("property", 5),
        ... )
        >>> positions_gradient = TensorBlock(
        ...     values=np.full((2, 3, 1, 5), 11.0),
        ...     samples=Labels(["sample", "atom"], np.array([[0, 2], [2, 3]])),
        ...     components=[
        ...         Labels.range("direction", 3),
        ...         Labels.range("component", 1),
        ...     ],
        ...     properties=Labels.range("property", 5),
        ... )

        >>> block.add_gradient("positions", positions_gradient)
        >>> cell_gradient = TensorBlock(
        ...     values=np.full((2, 3, 3, 1, 5), 15.0),
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

        >>> cell_gradient = block.gradient("cell")
        >>> print(cell_gradient)
        Gradient TensorBlock ('cell')
            samples (2): ['sample']
            components (3, 3, 1): ['direction_1', 'direction_2', 'component']
            properties (5): ['property']
            gradients: None
        """
        gradient_block = ctypes.POINTER(mts_block_t)()

        self._lib.mts_block_gradient(
            self.as_mts_block_t(), parameter.encode("utf8"), gradient_block
        )

        check_pointer(gradient_block)
        gradient = TensorBlock.unsafe_view_from_ptr(gradient_block, parent=self)

        gradient._gradient_parameters = copy.deepcopy(self._gradient_parameters)
        gradient._gradient_parameters.append(parameter)

        return gradient

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
        >>> from metatensor import Labels, TensorBlock
        >>> block = TensorBlock(
        ...     values=np.full((3, 1, 1), 1.0),
        ...     samples=Labels(["system"], np.array([[0], [2], [4]])),
        ...     components=[Labels.range("component", 1)],
        ...     properties=Labels.range("property", 1),
        ... )
        >>> gradient = TensorBlock(
        ...     values=np.full((2, 1, 1), 11.0),
        ...     samples=Labels(["sample", "parameter"], np.array([[0, -2], [2, 3]])),
        ...     components=[Labels.range("component", 1)],
        ...     properties=Labels.range("property", 1),
        ... )
        >>> block.add_gradient("parameter", gradient)
        >>> print(block)
        TensorBlock
            samples (3): ['system']
            components (1): ['component']
            properties (1): ['property']
            gradients: ['parameter']
        """
        if self._parent is not None:
            raise ValueError(
                "cannot add gradient on this block since it is a view inside "
                "a TensorMap or another TensorBlock"
            )

        self._lib.mts_block_add_gradient(
            self.as_mts_block_t(), parameter.encode("utf8"), gradient.release()
        )

    def gradients_list(self) -> List[str]:
        """get a list of all gradients defined in this block"""
        parameters = ctypes.POINTER(ctypes.c_char_p)()
        count = c_uintptr_t()
        self._lib.mts_block_gradients_list(self.as_mts_block_t(), parameters, count)

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

    @property
    def dtype(self) -> DType:
        """
        Get the dtype of all the values and gradient arrays stored inside this
        :py:class:`TensorBlock`.
        """
        if self._cached_dtype is None:
            self._cached_dtype = _data.array_dtype(self.values)
        return self._cached_dtype

    @property
    def device(self) -> Device:
        """
        Get the device of all the values and gradient arrays stored inside this
        :py:class:`TensorBlock`.
        """
        if self._cached_device is None:
            self._cached_device = _data.array_device(self.values)
        return self._cached_device

    @property
    def is_view(self) -> bool:
        """
        Check if this block is a view (i.e. does not own the underlying data).
        """
        return self._parent is not None

    def to(self, *args, **kwargs) -> "TensorBlock":
        """
        Move all the data in this block (labels, values, and gradients) to the given
        ``dtype``, ``device`` and ``arrays`` backend.

        :param dtype: new dtype to use for all arrays. The dtype stays the same if this
            is set to ``None``.
        :param device: new device to use for all arrays. The device stays the same if
            this is set to ``None``.
        :param Optional[str] arrays: new backend to use for the arrays. This can be
            either ``"numpy"``, ``"torch"`` or ``None`` (keeps the existing backend);
            and must be given as a keyword argument (``arrays="numpy"``).
        :param bool non_blocking: If this is ``True`` and the :py:class:`TensorBlock`
            contains ``"torch"`` arrays, the function tries to move the data
            asynchronously. See :py:meth:`torch.Tensor.to` for more information.
        """
        arrays = kwargs.pop("arrays", None)
        non_blocking = kwargs.pop("non_blocking", False)
        dtype, device = _data.to_arguments_parse("`TensorBlock.to`", *args, **kwargs)

        values = self.values

        if arrays is not None:
            values = _data.array_change_backend(values, arrays)

        if dtype is not None:
            values = _data.array_change_dtype(values, dtype, non_blocking=non_blocking)

        if device is not None:
            values = _data.array_change_device(
                values, device, non_blocking=non_blocking
            )

        block = TensorBlock(
            values,
            self.samples.to(device=device, arrays=arrays, non_blocking=non_blocking),
            [
                c.to(device=device, arrays=arrays, non_blocking=non_blocking)
                for c in self.components
            ],
            self.properties.to(device=device, arrays=arrays, non_blocking=non_blocking),
        )
        for parameter, gradient in self.gradients():
            block.add_gradient(
                parameter,
                gradient.to(
                    dtype=dtype,
                    device=device,
                    arrays=arrays,
                    non_blocking=non_blocking,
                ),
            )

        return block

    # ===== Serialization support ===== #

    @classmethod
    def _from_pickle(cls, buffer: Union[bytes, bytearray]):
        """
        Passed to pickler to reconstruct TensorBlock from bytes object
        """
        from .io import create_numpy_array, load_block_buffer_custom_array

        # TODO: make it so when saving data in torch tensors, we load back data in torch
        # tensors.
        return load_block_buffer_custom_array(buffer, create_numpy_array)

    def __reduce_ex__(self, protocol: int):
        """
        Used by the Pickler to dump TensorBlock object to bytes object. When protocol >=
        5 it supports PickleBuffer which reduces number of copies needed
        """
        from .io import _save_block_buffer_raw

        buffer = _save_block_buffer_raw(self)
        if protocol >= 5:
            return self._from_pickle, (PickleBuffer(buffer),)
        else:
            return self._from_pickle, (buffer.raw,)

    @staticmethod
    def load(
        file: Union[str, pathlib.Path, BinaryIO], use_numpy=False
    ) -> "TensorBlock":
        """
        Load a serialized :py:class:`TensorBlock` from a file or a buffer, calling
        :py:func:`metatensor.load_block`.

        :param file: file path or file object to load from
        :param use_numpy: should we use the numpy loader or metatensor's. See
            :py:func:`metatensor.load` for more information.
        """
        from .io import load_block

        return load_block(file=file, use_numpy=use_numpy)

    @staticmethod
    def load_buffer(
        buffer: Union[bytes, bytearray, memoryview],
        use_numpy=False,
    ) -> "TensorBlock":
        """
        Load a serialized :py:class:`TensorMap` from a buffer, calling
        :py:func:`metatensor.io.load_block_buffer`.

        :param buffer: in-memory buffer containing the data
        :param use_numpy: should we use the numpy loader or metatensor's. See
            :py:func:`metatensor.load` for more information.
        """
        from .io import load_block_buffer

        return load_block_buffer(buffer=buffer)

    def save(self, file: Union[str, pathlib.Path, BinaryIO], use_numpy=False):
        """
        Save this :py:class:`TensorBlock` to a file or a buffer, calling
        :py:func:`metatensor.save`.

        :param file: file path or file object to save to
        :param use_numpy: should we use the numpy serializer or metatensor's. See
            :py:func:`metatensor.save` for more information.
        """
        from .io import save

        return save(file=file, data=self, use_numpy=use_numpy)

    def save_buffer(self, use_numpy=False) -> memoryview:
        """
        Save this :py:class:`TensorBlock` to an in-memory buffer, calling
        :py:func:`metatensor.io.save_buffer`.

        :param use_numpy: should we use numpy serialization or metatensor's. See
            :py:func:`metatensor.save` for more information.
        """
        from .io import save_buffer

        return save_buffer(data=self, use_numpy=use_numpy)
