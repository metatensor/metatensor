import torch

from .._backend import isinstance_metatensor


class Module(torch.nn.Module):
    """
    This class should be used instead of :py:class:`torch.nn.Module` when your module
    contains data stored inside :py:class:`metatensor.torch.Labels`,
    :py:class:`metatensor.torch.TensorBlock` or :py:class:`metatensor.torch.TensorMap`.

    It ensures that this data is properly moved to other dtype and devices when calling
    ``.to()``, ``.cuda()``, ``.float()`` and other related functions.

    We support storing these class either directly as attributes (``self.name = ...``),
    or inside arbitrarily nested dict, list, or tuple (``self.name = {"dict": [...]}``).

    Below is an example creating a custom linear model, that stores the output
    ``properties`` as an attribute. The corresponding labels will automatically be moved
    on device at the same time as the module.

    >>> import torch
    >>> from typing import List
    >>> from metatensor.torch.learn import nn
    >>> from metatensor.torch import Labels, TensorMap, TensorBlock
    >>>
    >>> class CustomLinear(nn.Module):
    ...     def __init__(self, in_features, out_features):
    ...         super().__init__()
    ...         self.properties = Labels(
    ...             ["out_features"], torch.arange(out_features).reshape(-1, 1)
    ...         )
    ...         self.linear = torch.nn.Linear(in_features, out_features)
    ...
    ...     def forward(self, tensor: TensorMap) -> TensorMap:
    ...         blocks: List[TensorBlock] = []
    ...         for block in tensor:
    ...             new_values = self.linear(block.values)
    ...             new_block = TensorBlock(
    ...                 values,
    ...                 block.samples,
    ...                 block.components,
    ...                 self.properties,
    ...             )
    ...             blocks.append(new_block)
    ...         return TensorBlock(tensor.keys, blocks)
    """

    def __init__(self):
        """"""
        super().__init__()
        self.register_buffer("_mts_helper", torch.zeros(0))

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
