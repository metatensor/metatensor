from typing import List, Optional

import torch
from torch.nn import Module

from .._backend import Labels, TensorMap
from .module_map import ModuleMap


class Tanh(Module):
    """
    Module similar to :py:class:`torch.nn.Tanh` that works with
    :py:class:`metatensor.torch.TensorMap` objects.

    Applies a hyperbolic tangent transformation to each block of a :py:class:`TensorMap`
    passed to its forward method, indexed by :param in_keys:.

    Refer to the :py:class`torch.nn.Tanh` documentation for a more detailed description
    of the parameters.

    :param in_keys: :py:class:`Labels`, the keys that are assumed to be in the input
        tensor map in the :py:meth:`forward` method.
    :param out_properties: list of :py:class`Labels` (optional), the properties labels
        of the output. By default the output properties are relabeled using
        Labels.range.
    """

    def __init__(
        self, in_keys: Labels, out_properties: Optional[Labels] = None
    ) -> None:
        super().__init__()
        modules: List[Module] = [torch.nn.Tanh() for i in range(len(in_keys))]
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


class InvariantTanh(torch.nn.Module):
    """
    Module similar to :py:class:`torch.nn.Tanh` that works with
    :py:class:`metatensor.torch.TensorMap` objects, applying the transformation only to
    the invariant blocks.

    Applies a hyperbolic tangent transformation to each invariant block of a
    :py:class:`TensorMap` passed to its :py:meth:`forward` method. These are indexed by
    the keys in :param in_keys: at numeric indices passed in :param invariant_key_idxs:.

    Refer to the :py:class`torch.nn.Tanh` documentation for a more detailed description
    of the parameters.

    :param in_keys: :py:class:`Labels`, the keys that are assumed to be in the input
        tensor map in the :py:meth:`forward` method.
    :param invariant_key_idxs: list of int, the indices of the invariant keys present in
        `in_keys` in the input :py:class:`TensorMap`. Only blocks for these keys will
        have the tanh transformation applied. Covariant blocks will have the identity
        operator applied.
    :param out_properties: list of :py:class`Labels` (optional), the properties labels
        of the output. By default the output properties are relabeled using
        Labels.range.
    """

    def __init__(
        self,
        in_keys: Labels,
        invariant_key_idxs: List[int],
        out_properties: Optional[Labels] = None,
    ) -> None:
        super().__init__()
        modules: List[Module] = []
        for i in range(len(in_keys)):
            if i in invariant_key_idxs:  # Invariant block: apply tanh
                module = torch.nn.Tanh()
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
