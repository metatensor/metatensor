import numpy as np
import torch

from .._backend import Labels, TensorBlock, TensorMap, isinstance_metatensor


class Module(torch.nn.Module):
    """
    This class should be used instead of :py:class:`torch.nn.Module` when your module
    contains data stored inside :py:class:`metatensor.torch.Labels`,
    :py:class:`metatensor.torch.TensorBlock` or :py:class:`metatensor.torch.TensorMap`.

    It ensures that this data is properly moved to other dtype and devices when calling
    ``.to()``, ``.cuda()``, ``.float()`` and other related functions. We also handle the
    corresponding data in ``state_dict()`` and ``load_state_dict()``.

    We support storing these class either directly as attributes (``self.name = ...``),
    or inside arbitrarily nested dict, list, or tuple (``self.name = {"dict": [...]}``).
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("_mts_helper", torch.zeros(0))

        # register state dict hooks to handle metatensor data. This is done through
        # hooks instead of custom `get_extra_state`/`set_extra_state` to handle
        # submodules that contain metatensor data but are still using `torch.nn.Module`
        # as a base class
        self.register_state_dict_post_hook(_state_dict_post_hook)
        self.register_load_state_dict_pre_hook(_load_state_dict_pre_hook)

    # _apply is used to define all of the data movement functions (`to()`, `cuda()`,
    # `double()`, …), so we only need to override this one.
    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # the _apply call should have moved everything to the correct device/dtype,
        # including our `_mts_helper` buffer. Now we can move all metatensor data to
        # the same device/dtype
        device = self._mts_helper.device
        dtype = self._mts_helper.dtype

        for name, value in self.__dict__.items():
            value, changed = _metatensor_data_to(value, dtype=dtype, device=device)
            if changed:
                self.__dict__[name] = value

        return result


# WARNING: used by metatensor.torch.learn, do not update without increasing the version
# number as for a breaking change.
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
        updated = []
        some_changed = False
        for tuple_value in value:
            updated_value, changed = _metatensor_data_to(tuple_value, dtype, device)
            some_changed = some_changed or changed
            updated.append(updated_value)

        if some_changed:
            return tuple(updated), True

    return value, False


def _state_dict_post_hook(module, state_dict, prefix, local_metadata):
    for name, value in module.__dict__.items():
        serialized_value, needs_storing = _serialize_metatensor(value)
        if needs_storing:
            state_dict[f"{prefix}{name}"] = serialized_value


def _serialize_metatensor(value):
    """TODO"""
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
        serialized = {}
        all_need_storing = True
        some_need_storing = False
        for key, dict_value in value.items():
            serialized_value, needs_storing = _serialize_metatensor(dict_value)
            all_need_storing = all_need_storing and needs_storing
            some_need_storing = some_need_storing or needs_storing
            serialized[key] = serialized_value

        if some_need_storing:
            if not all_need_storing:
                # we got some unexpected type somewhere
                raise ValueError("TODO")
            return serialized, True

    elif isinstance(value, list):
        serialized = []
        all_need_storing = True
        some_need_storing = False
        for list_value in value:
            serialized_value, needs_storing = _serialize_metatensor(list_value)
            all_need_storing = all_need_storing and needs_storing
            some_need_storing = some_need_storing or needs_storing
            serialized.append(serialized_value)

        if some_need_storing:
            if not all_need_storing:
                # we got some unexpected type somewhere
                raise ValueError("TODO")
            return serialized, True

    elif isinstance(value, tuple):
        serialized = []
        some_need_storing = False
        for tuple_value in value:
            serialized_value, needs_storing = _serialize_metatensor(tuple_value)
            some_need_storing = some_need_storing or needs_storing
            serialized.append(serialized_value)

        if some_need_storing:
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


def _load_state_dict_pre_hook(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    used_keys = []
    for name, value in state_dict.items():
        deserialized_value, needs_storing = _deserialize_metatensor(value)
        if needs_storing:
            module.__setattr__(name[len(prefix) :], deserialized_value)
            used_keys.append(name)

    for key in used_keys:
        del state_dict[key]


def _deserialize_metatensor(value):
    if (
        isinstance(value, tuple)
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
        deserialized = {}
        all_need_storing = True
        some_need_storing = False
        for name, dict_value in value.items():
            deserialized_value, needs_storing = _deserialize_metatensor(dict_value)
            all_need_storing = all_need_storing and needs_storing
            some_need_storing = some_need_storing or needs_storing
            deserialized[name] = deserialized_value

        if some_need_storing:
            if not all_need_storing:
                # we got some unexpected type somewhere
                raise ValueError("TODO")
            return deserialized, True

    elif isinstance(value, list):
        deserialized = []
        all_need_storing = True
        some_need_storing = False
        for list_value in value:
            deserialized_value, needs_storing = _deserialize_metatensor(list_value)
            all_need_storing = all_need_storing and needs_storing
            some_need_storing = some_need_storing or needs_storing
            deserialized.append(deserialized_value)

        if some_need_storing:
            if not all_need_storing:
                # we got some unexpected type somewhere
                raise ValueError("TODO")
            return deserialized, True

    elif isinstance(value, tuple):
        deserialized = []
        some_need_storing = False
        for tuple_value in value:
            deserialized_value, needs_storing = _deserialize_metatensor(tuple_value)
            some_need_storing = some_need_storing or needs_storing
            deserialized.append(deserialized_value)

        if some_need_storing:
            return tuple(deserialized), True

    return None, False
