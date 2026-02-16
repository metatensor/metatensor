from typing import Union

import torch

from metatensor.torch import Labels, TensorBlock, TensorMap


def isinstance_metatensor(value: Union[Labels, TensorBlock, TensorMap], typename: str):
    assert typename in ("Labels", "TensorBlock", "TensorMap")

    if torch.jit.is_scripting():
        if typename == "Labels":
            return isinstance(value, Labels)
        elif typename == "TensorBlock":
            return isinstance(value, TensorBlock)
        elif typename == "TensorMap":
            return isinstance(value, TensorMap)

    # For custom classes (TensorMap, …), the `values` is an instance of
    # `torch.ScriptObject` and not the class itself, so we use `_type`
    # to get the type name
    if isinstance(value, torch.ScriptObject):
        qualified_name = value._type().qualified_name()
        if qualified_name.startswith("__torch__.torch.classes.metatensor"):
            return value._type().name() == typename

    return False


# WARNING: this is duplicated in metatensor.learn.nn._module, make sure to change both
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


def _apply_metatensor(module):
    if "_mts_helper" in module._buffers:
        device = module._mts_helper.device
        dtype = module._mts_helper.dtype

        if isinstance(module, torch.jit.RecursiveScriptModule):
            # Get the list of attributes for this module.
            #
            # Unfortunately `named_attributes` is not available to Python, so we need to
            # parse the list from a string.
            attributes = module._c.dump_to_str(code=False, attrs=True, params=False)
            attributes = _parse_rsm_attributes(attributes)

            # Update the attributes and re-add them to the module with
            # `_register_attribute` (which does an update when the attribute already
            # exists.)
            for name in attributes:
                value = module._c.getattr(name)

                value, changed = _metatensor_data_to(value, dtype=dtype, device=device)
                if changed:
                    typ = _get_torch_type(value)
                    module._c._register_attribute(name, typ, value)


def _get_torch_type(value):
    if isinstance(value, torch.ScriptObject):
        return value._type()
    elif isinstance(value, int):
        return torch._C.IntType.get()
    elif isinstance(value, float):
        return torch._C.FloatType.get()
    elif isinstance(value, str):
        return torch._C.StringType.get()
    elif isinstance(value, bool):
        return torch._C.BoolType.get()
    elif isinstance(value, torch.Tensor):
        return torch._C.TensorType.get()
    elif isinstance(value, dict):
        # assume that all keys/values have the same type, TorchScript would enforce it
        # anyway
        key, value = next(iter(value.items()))
        return torch._C.DictType(_get_torch_type(key), _get_torch_type(value))
    elif isinstance(value, list):
        # assume that all values have the same type, TorchScript would enforce it
        # anyway
        value = next(iter(value))
        return torch._C.ListType(_get_torch_type(value))
    elif isinstance(value, tuple):
        return torch._C.TupleType([_get_torch_type(v) for v in value])
    else:
        return None


def _parse_rsm_attributes(string):
    """
    Parse the output of ``ScriptModule.dump_to_str()`` to extract the list of attributes
    on a module.
    """
    attributes = []
    in_attributes = False
    for line in string.splitlines():
        if not in_attributes:
            if "attributes {" in line:
                in_attributes = True
                continue

        else:
            splited = line.split("=")
            if len(splited) == 2:
                name = splited[0].strip()
                attributes.append(name)
            elif line.endswith("}"):
                # we are done
                return attributes
            else:
                raise RuntimeError(f"failed to parse line in attributes: '{line}'")

    raise RuntimeError(f"failed to parse attributes section in:\n{string}")


def _patch_torch_jit_module():
    """
    Monkey-patch `torch.jit.RecursiveScriptModule._apply` to also handle metatensor
    data, similar to what's happening in `metatensor.learn.nn.Module._apply`
    """

    _original_rsm_apply = torch.jit.RecursiveScriptModule._apply

    def _new_rsm_apply(self, fn, recurse=True):
        output = _original_rsm_apply(self, fn, recurse=recurse)
        _apply_metatensor(self)

        if recurse:
            for module in self.children():
                _apply_metatensor(module)

        return output

    torch.jit.RecursiveScriptModule._apply = _new_rsm_apply
