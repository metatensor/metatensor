from typing import List

import torch
from torch.nn import Module

from .._backend import Labels, TensorMap, is_metatensor_class
from .module_map import ModuleMap


class Sequential(Module):
    """
    A sequential model that applies a list of ModuleMaps to the input in order.

    :param in_keys:
        The keys that are assumed to be in the input tensor map in the
        :py:meth:`forward` function.
    :param args:
        A list of :py:class:`ModuleMap` objects that will be applied in order to
        the input tensor map in the :py:meth:`forward` function.
    """

    def __init__(self, in_keys: Labels, *args: List[ModuleMap]):
        super().__init__()
        if not is_metatensor_class(in_keys, Labels):
            raise TypeError("`in_keys` must be a `Labels` object.")

        modules: List[Module] = []
        for i in range(len(in_keys)):
            modules.append(torch.nn.Sequential(*[arg.module_map[i] for arg in args]))
        self.module_map = ModuleMap(
            in_keys, modules, out_properties=args[-1].module_map.out_properties
        )

    def forward(self, tensor: TensorMap) -> TensorMap:
        """
        Apply the transformations to the input tensor map `tensor`.
        """
        return self.module_map(tensor)
