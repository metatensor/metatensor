from typing import List, Optional

import torch
from torch.nn import Module

from .._backend import Labels, TensorMap
from .._dispatch import int_array_like
from .module_map import ModuleMap


class ReLU(Module):
    """
    Module similar to :py:class:`torch.nn.ReLU` that works with
    :py:class:`metatensor.torch.TensorMap` objects.

    Applies a rectified linear unit transformation transformation to each block of a
    :py:class:`TensorMap` passed to its forward method, indexed by :param in_keys:.

    Refer to the :py:class`torch.nn.ReLU` documentation for a more detailed description
    of the parameters.

    :param in_keys: :py:class:`Labels`, the keys that are assumed to be in the input
        tensor map in the :py:meth:`forward` method.
    :param out_properties: list of :py:class`Labels` (optional), the properties labels
        of the output. By default the output properties are relabeled using
        Labels.range.
    """

    def __init__(
        self,
        in_keys: Labels,
        out_properties: Optional[Labels] = None,
        *,
        in_place: bool = False,
    ) -> None:
        super().__init__()
        modules: List[Module] = [torch.nn.ReLU(in_place) for i in range(len(in_keys))]
        self.module_map = ModuleMap(in_keys, modules, out_properties)

    def forward(self, tensor: TensorMap) -> TensorMap:
        """
        Apply the transformation to the input tensor map `tensor`.

        Note: currently not supporting gradients.

        :param tensor: :py:class:`TensorMap` with the input tensor to be transformed.
        :return: :py:class:`TensorMap`
        """
        # Currently not supporting gradients
        if len(tensor[0].gradients_list()) != 0:
            raise ValueError(
                "Gradients not supported. Please use metatensor.remove_gradients()"
                " before using this module"
            )
        return self.module_map(tensor)


class InvariantReLU(torch.nn.Module):
    """
    Module similar to :py:class:`torch.nn.ReLU` that works with
    :py:class:`metatensor.torch.TensorMap` objects, applying the transformation only to
    the invariant blocks.

    Applies a rectified linear unit transformation to each invariant block of a
    :py:class:`TensorMap` passed to its :py:meth:`forward` method. These are indexed by
    the keys in :param in_keys: that correspond to the selection passed in :param
    invariant_keys:.

    Refer to the :py:class`torch.nn.ReLU` documentation for a more detailed description
    of the parameters.

    :param in_keys: :py:class:`Labels`, the keys that are assumed to be in the input
        tensor map in the :py:meth:`forward` method.
    :param out_properties: list of :py:class`Labels` (optional), the properties labels
        of the output. By default the output properties are relabeled using
        Labels.range.
    :param invariant_keys: a :py:class:`Labels` object that is used to select the
        invariant keys from ``in_keys``. If not provided, the invariant keys are assumed
        to be those where key dimensions ``["o3_lambda", "o3_sigma"]`` are equal to
        ``[0, 1]``.
    """

    def __init__(
        self,
        in_keys: Labels,
        out_properties: Optional[Labels] = None,
        invariant_keys: Optional[Labels] = None,
        *,
        in_place: bool = False,
    ) -> None:
        super().__init__()
        # Set a default for invariant keys
        if invariant_keys is None:
            invariant_keys = Labels(
                names=["o3_lambda", "o3_sigma"],
                values=int_array_like([0, 1], like=in_keys.values).reshape(-1, 2),
            )
        invariant_key_idxs = in_keys.select(invariant_keys)
        modules: List[Module] = []
        for i in range(len(in_keys)):
            if i in invariant_key_idxs:  # Invariant block: apply ReLU
                module = torch.nn.ReLU(in_place)
            else:  # Covariant block: apply identity operator
                module = torch.nn.Identity()
            modules.append(module)
        self.module_map: ModuleMap = ModuleMap(in_keys, modules, out_properties)

    def forward(self, tensor: TensorMap) -> TensorMap:
        """
        Apply the transformation to the input tensor map `tensor`.

        Note: currently not supporting gradients.

        :param tensor: :py:class:`TensorMap` with the input tensor to be transformed.
        :return: :py:class:`TensorMap`
        """
        # Currently not supporting gradients
        if len(tensor[0].gradients_list()) != 0:
            raise ValueError(
                "Gradients not supported. Please use metatensor.remove_gradients()"
                " before using this module"
            )
        return self.module_map(tensor)
