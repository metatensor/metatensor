from typing import List

import numpy as np
import torch

from .._backend import Labels, TensorBlock, TensorMap, isinstance_metatensor


class Buffer:
    """
    Wrapper for metatensor data, similar to :py:class:`torch.nn.Buffer` but for
    metatensor types (Labels, TensorBlock, TensorMap) and (possibly nested)
    dict/list/tuple containers of metatensor types.

    When assigned to a :py:class:`Module` attribute in ``__init__``, the value
    is automatically registered via :py:meth:`Module.register_buffer` and will
    be properly handled by ``.to()`` and ``state_dict()``.

    See :ref:`learn-tutorial-nn-custom-module` for a tutorial covering this
    class.

    :param value: the metatensor value to wrap
    :param persistent: whether the buffer is part of the module's
        ``state_dict``, passed through to
        :py:meth:`Module.register_buffer`
    """

    def __init__(self, value, persistent=True):
        self.data = value
        self.persistent = persistent


class Module(torch.nn.Module):
    """
    This class should be used instead of :py:class:`torch.nn.Module` when your module
    contains data stored inside :py:class:`metatensor.torch.Labels`,
    :py:class:`metatensor.torch.TensorBlock` or :py:class:`metatensor.torch.TensorMap`.

    It ensures that this data is properly moved to other dtype and devices when calling
    ``.to()``, ``.cuda()``, ``.float()`` and other related functions. We also handle the
    corresponding data in ``state_dict()`` and ``load_state_dict()``.

    Members containing metatensor data should be registered via
    ``self.register_buffer(name, value)``. This supports plain metatensor values as well
    as arbitrarily nested dict, list, or tuple of metatensor values.

    Below is an example creating a custom linear model, that stores the output
    ``properties`` as a registered buffer. The corresponding labels will automatically
    be moved on device at the same time as the module, and will be included in the state
    dict.

    >>> import torch
    >>> from typing import List
    >>> from metatensor.torch.learn import nn
    >>> from metatensor.torch import Labels, TensorMap, TensorBlock
    >>>
    >>> class CustomLinear(nn.Module):
    ...     def __init__(self, in_features: int, out_features: int):
    ...         super().__init__()
    ...         self.register_buffer(
    ...             "properties",
    ...             Labels("out_features", torch.arange(out_features).reshape(-1, 1)),
    ...         )
    ...         self.linear = torch.nn.Linear(in_features, out_features)
    ...
    ...     def forward(self, tensor: TensorMap) -> TensorMap:
    ...         blocks: List[TensorBlock] = []
    ...         for block in tensor:
    ...             new_values = self.linear(block.values)
    ...             new_block = TensorBlock(
    ...                 new_values,
    ...                 block.samples,
    ...                 block.components,
    ...                 self.properties,
    ...             )
    ...         blocks.append(new_block)
    ...         return TensorMap(tensor.keys, blocks)

    See :ref:`learn-tutorial-nn-custom-module` for a tutorial covering this
    class, including how it differs from :py:class:`ModuleMap`.
    """

    _mts_buffer_names: List[str]
    _mts_non_persistent_buffers: List[str]

    def __init__(self):
        """"""
        super().__init__()
        self._mts_buffer_names = []
        self._mts_non_persistent_buffers = []
        self.register_buffer("_mts_helper", torch.zeros(0))

    def __setattr__(self, name, value):
        if isinstance(value, Buffer):
            self.register_buffer(name, value.data, persistent=value.persistent)
        else:
            super().__setattr__(name, value)

    def register_buffer(self, name, value, persistent=True):
        """
        Register a buffer containing metatensor data on this module, or register a
        regular PyTorch tensor buffer.

        If ``value`` is a :py:class:`torch.Tensor`, this is the same as
        :py:meth:`torch.nn.Module.register_buffer`.

        If ``value`` is one of the metatensor types (Labels, TensorBlock, TensorMap) or
        a (possibly nested) dict/list/tuple containing metatensor types, it will be
        stored on the module and properly handled by ``.to()`` and ``state_dict()``.

        :param name: name of the buffer
        :param value: value to register
        :param persistent: whether the buffer is part of the module's ``state_dict``. If
            this is ``False``, the buffer will still be moved to the correct
            device/dtype when calling ``.to()``.
        """
        if isinstance(value, Buffer):
            value = value.data

        if isinstance(value, torch.Tensor):
            super().register_buffer(name, value, persistent=persistent)
            return

        if value is not None and not _contains_metatensor(value):
            # Allow None (optional placeholder) and empty containers
            # (runtime-configurable), but reject other non-metatensor types
            if isinstance(value, (dict, list, tuple)) and len(value) == 0:
                pass
            else:
                raise TypeError(
                    f"register_buffer expects a torch.Tensor, None, or a value "
                    f"containing metatensor types (Labels, TensorBlock, TensorMap), "
                    f"got {type(value).__name__}"
                )

        if name in self._mts_buffer_names:
            raise KeyError(f"buffer '{name}' is already registered")

        self._mts_buffer_names.append(name)
        if not persistent:
            self._mts_non_persistent_buffers.append(name)

        self.__setattr__(name, value)

    def get_extra_state(self):
        extra = {}
        for name in self._mts_buffer_names:
            if name in self._mts_non_persistent_buffers:
                continue
            serialized_value, needs_storing = _serialize_metatensor(getattr(self, name))
            if needs_storing:
                extra[name] = serialized_value
        return extra

    def set_extra_state(self, extra):
        for name, value in extra.items():
            deserialized_value, needs_storing = _deserialize_metatensor(value)
            if needs_storing:
                self.__setattr__(name, deserialized_value)

    # _apply is used to define all of the data movement functions (`to()`, `cuda()`,
    # `double()`, …), so we only need to override this one.
    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # the _apply call should have moved everything to the correct device/dtype,
        # including our `_mts_helper` buffer. Now we can move all metatensor data to
        # the same device/dtype
        device = self._mts_helper.device
        dtype = self._mts_helper.dtype

        for name in self._mts_buffer_names:
            value = getattr(self, name)
            value, changed = _metatensor_data_to(value, dtype=dtype, device=device)
            if changed:
                self.__setattr__(name, value)

        return result


def _contains_metatensor(value):
    """
    Check if ``value`` is a metatensor type (Labels, TensorBlock, TensorMap) or
    recursively contains one inside dict, list, or tuple.
    """
    if (
        isinstance_metatensor(value, "Labels")
        or isinstance_metatensor(value, "TensorBlock")
        or isinstance_metatensor(value, "TensorMap")
    ):
        return True
    elif isinstance(value, dict):
        return any(_contains_metatensor(v) for v in value.values())
    elif isinstance(value, (list, tuple)):
        return any(_contains_metatensor(v) for v in value)
    return False


# WARNING: this is duplicated in metatensor.torch._module, make sure to change both
# versions of the function at the same time
def _metatensor_data_to(value, dtype, device):
    """
    Convert metatensor data to the given dtype/device, returning the new data and a
    bool to indicate if the value was modified.
    """
    if isinstance_metatensor(value, "Labels"):
        return value.to(device=device), True
    elif isinstance_metatensor(value, "TensorBlock"):
        return value.to(device=device, dtype=dtype), True
    elif isinstance_metatensor(value, "TensorMap"):
        return value.to(device=device, dtype=dtype), True
    elif isinstance(value, dict):
        if len(value) == 0:
            return value, True

        updated = {}
        all_changed = True
        some_changed = False
        for name, dict_value in value.items():
            updated_value, changed = _metatensor_data_to(dict_value, dtype, device)
            all_changed = all_changed and changed
            some_changed = some_changed or changed
            updated[name] = updated_value

        if some_changed:
            if not all_changed:
                # we got some unexpected type somewhere
                raise ValueError(
                    "dicts containing both metatensor and non-metatensor data as "
                    "values are not supported"
                )
            return updated, True

    elif isinstance(value, list):
        if len(value) == 0:
            return value, True

        updated = []
        all_changed = True
        some_changed = False
        for list_value in value:
            updated_value, changed = _metatensor_data_to(list_value, dtype, device)
            all_changed = all_changed and changed
            some_changed = some_changed or changed
            updated.append(updated_value)

        if some_changed:
            if not all_changed:
                # we got some unexpected type somewhere
                raise ValueError(
                    "lists containing both metatensor and non-metatensor data "
                    "are not supported"
                )
            return updated, True

    elif isinstance(value, tuple):
        if len(value) == 0:
            return value, True

        updated = []
        some_changed = False
        for tuple_value in value:
            updated_value, changed = _metatensor_data_to(tuple_value, dtype, device)
            some_changed = some_changed or changed
            updated.append(updated_value)

        if some_changed:
            return tuple(updated), True

    return value, False


def _serialize_metatensor(value):
    """
    If ``value`` is an instance of one of the metatensor classes (Labels, TensorBlock,
    TensorMap), transform it into a format that can more easily be used in a PyTorch
    state dict.

    This format is a tuple with three elements: ``(class_name, buffer, tensor)``, where
    the class name is a string (``"metatensor.Labels"``, ``"metatensor.TensorBlock"``,
    or ``"metatensor.TensorMap"``); the buffer is the result of
    ``metatensor.save_buffer``; and the tensor is an empty tensor carrying information
    about the origin device and dtype of the data.

    If ``value`` is an (arbitrarily nested) container with metatensor data (i.e.
    ``Dict[..., Labels]``, ``List[Tuple[Labels, TensorBlock]]``, etc.), then the
    metatensor data is replaced in place with the data described above.

    This function then returns the serialized data, and a boolean indicating if
    ``value`` contained metatensor data at any point.
    """
    if isinstance_metatensor(value, "Labels"):
        serialized = (
            "metatensor.Labels",
            _buffer_to_picklable(value.save_buffer()),
            _empty_tensor_like(value.values),
        )
        return serialized, True

    elif isinstance_metatensor(value, "TensorBlock"):
        # we only support torch Tensors as the block values
        assert isinstance(value.values, torch.Tensor), (
            "data must be stored in torch Tensors"
        )
        serialized = (
            "metatensor.TensorBlock",
            # until we have https://github.com/metatensor/metatensor/issues/775,
            # move everything to CPU/float64
            _buffer_to_picklable(
                value.to(device="cpu").to(dtype=torch.float64).save_buffer()
            ),
            _empty_tensor_like(value.values),
        )
        return serialized, True

    elif isinstance_metatensor(value, "TensorMap"):
        if len(value) == 0:
            empty_tensor = _empty_tensor_like(value.keys.values)
        else:
            # we only support torch Tensors as the block values
            assert isinstance(value.block(0).values, torch.Tensor), (
                "data must be stored in torch Tensors"
            )
            empty_tensor = _empty_tensor_like(value.block(0).values)

        serialized = (
            "metatensor.TensorMap",
            # same as above
            _buffer_to_picklable(
                value.to(device="cpu").to(dtype=torch.float64).save_buffer()
            ),
            empty_tensor,
        )
        return serialized, True

    elif isinstance(value, dict):
        if len(value) == 0:
            return value, True

        serialized = {}
        all_contains_mts = True
        some_contains_mts = False
        for key, dict_value in value.items():
            serialized_value, contains_mts = _serialize_metatensor(dict_value)
            all_contains_mts = all_contains_mts and contains_mts
            some_contains_mts = some_contains_mts or contains_mts
            serialized[key] = serialized_value

        if some_contains_mts:
            if not all_contains_mts:
                # we got some unexpected type somewhere
                raise TypeError(
                    "Dict containing both metatensor and other data together "
                    "are not supported"
                )
            return serialized, True

    elif isinstance(value, list):
        if len(value) == 0:
            return value, True

        serialized = []
        all_contains_mts = True
        some_contains_mts = False
        for list_value in value:
            serialized_value, contains_mts = _serialize_metatensor(list_value)
            all_contains_mts = all_contains_mts and contains_mts
            some_contains_mts = some_contains_mts or contains_mts
            serialized.append(serialized_value)

        if some_contains_mts:
            if not all_contains_mts:
                # we got some unexpected type somewhere
                raise TypeError(
                    "List containing both metatensor and other data together "
                    "are not supported"
                )
            return serialized, True

    elif isinstance(value, tuple):
        if len(value) == 0:
            return value, True

        serialized = []
        some_contains_mts = False
        for tuple_value in value:
            serialized_value, contains_mts = _serialize_metatensor(tuple_value)
            some_contains_mts = some_contains_mts or contains_mts
            serialized.append(serialized_value)

        if some_contains_mts:
            return tuple(serialized), True

    return None, False


def _empty_tensor_like(value):
    """Get a tensor with no data and the same dtype/device as ``value``"""
    if isinstance(value, torch.Tensor):
        return torch.empty(0, dtype=value.dtype, device=value.device)
    else:
        assert isinstance(value, np.ndarray)
        array = np.empty(0, dtype=value.dtype)
        return torch.from_numpy(array)


def _buffer_to_picklable(buffer):
    """
    Convert the buffer type used in metatensor-core (memoryview) to something pickle can
    handle while leaving the buffer type in metatensor-torch (torch.Tensor) alone
    """
    if isinstance(buffer, torch.Tensor):
        return buffer
    else:
        return buffer.tobytes()


def _deserialize_metatensor(value):
    """
    This function does the inverse of ``_serialize_metatensor``, re-creating metatensor
    data from the tuple representation.

    The function returns the deserialized data, and a boolean indicating if ``value``
    contains some metatensor data somewhere.
    """

    if (
        isinstance(value, tuple)
        and len(value) == 3
        and isinstance(value[0], str)
        and value[0].startswith("metatensor.")
    ):
        class_name, buffer, dtype_device_tensor = value

        if isinstance(buffer, torch.Tensor):
            buffer = buffer.to(device="cpu")

        if class_name == "metatensor.Labels":
            labels = Labels.load_buffer(buffer).to(dtype_device_tensor.device)
            return labels, True

        elif class_name == "metatensor.TensorBlock":
            block = TensorBlock.load_buffer(buffer)
            block = block.to(
                dtype=dtype_device_tensor.dtype,
                device=dtype_device_tensor.device,
                arrays="torch",
            )
            return block, True

        elif class_name == "metatensor.TensorMap":
            tensor = TensorMap.load_buffer(buffer)
            tensor = tensor.to(
                dtype=dtype_device_tensor.dtype,
                device=dtype_device_tensor.device,
                arrays="torch",
            )
            return tensor, True

        else:
            raise ValueError(f"got unexpected class name: '{class_name}'")

    elif isinstance(value, dict):
        if len(value) == 0:
            return value, True

        deserialized = {}
        all_contains_mts = True
        some_contains_mts = False
        for name, dict_value in value.items():
            deserialized_value, contains_mts = _deserialize_metatensor(dict_value)
            all_contains_mts = all_contains_mts and contains_mts
            some_contains_mts = some_contains_mts or contains_mts
            deserialized[name] = deserialized_value

        if some_contains_mts:
            if not all_contains_mts:
                # we got some unexpected type somewhere
                raise TypeError(
                    "Dict containing both metatensor and other data together "
                    "are not supported"
                )
            return deserialized, True

    elif isinstance(value, list):
        if len(value) == 0:
            return value, True

        deserialized = []
        all_contains_mts = True
        some_contains_mts = False
        for list_value in value:
            deserialized_value, contains_mts = _deserialize_metatensor(list_value)
            all_contains_mts = all_contains_mts and contains_mts
            some_contains_mts = some_contains_mts or contains_mts
            deserialized.append(deserialized_value)

        if some_contains_mts:
            if not all_contains_mts:
                # we got some unexpected type somewhere
                raise TypeError(
                    "List containing both metatensor and other data together "
                    "are not supported"
                )
            return deserialized, True

    elif isinstance(value, tuple):
        if len(value) == 0:
            return value, True

        deserialized = []
        some_contains_mts = False
        for tuple_value in value:
            deserialized_value, contains_mts = _deserialize_metatensor(tuple_value)
            some_contains_mts = some_contains_mts or contains_mts
            deserialized.append(deserialized_value)

        if some_contains_mts:
            return tuple(deserialized), True

    return None, False
