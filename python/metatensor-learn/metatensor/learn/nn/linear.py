from typing import List, Optional, Union

import torch
from torch.nn import Module

from .._backend import Labels, TensorMap
from ._utils import _check_module_map_parameter
from .module_map import ModuleMap


class Linear(Module):
    """
    Module similar to :py:class:`torch.nn.Linear` that works with
    :py:class:`metatensor.torch.TensorMap`.

    Applies a linear transformation to each block of a :py:class:`TensorMap` passed to
    its forward method, indexed by :param in_keys:.

    Refer to the :py:class`torch.nn.Linear` documentation for a more detailed
    description of the other parameters.

    Each parameter can be passed as a single value of its expected type, which is used
    as the parameter for all blocks. Alternatively, they can be passed as a list to
    control the parameters applied to each block indexed by the keys in :param in_keys:.

    :param in_keys: :py:class:`Labels`, the keys that are assumed to be in the input
        tensor map in the :py:meth:`forward` method.
    :param out_properties: list of :py:class`Labels` (optional), the properties labels
        of the output. By default the output properties are relabeled using
        Labels.range. If provided, :param out_features: can be inferred and need not be
        provided.
    """

    def __init__(
        self,
        in_keys: Labels,
        in_features: Union[int, List[int]],
        out_features: Optional[Union[int, List[int]]] = None,
        out_properties: Optional[List[Labels]] = None,
        *,
        bias: Union[bool, List[bool]] = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # Infer `out_features` if not provided
        if out_features is None:
            if out_properties is None:
                raise ValueError(
                    "If `out_features` is not provided,"
                    " `out_properties` must be provided."
                )
            out_features = [len(p) for p in out_properties]

        # Check input parameters, convert to lists (for each key) if necessary
        in_features = _check_module_map_parameter(
            in_features, "in_features", int, len(in_keys), "in_keys"
        )
        out_features = _check_module_map_parameter(
            out_features, "out_features", int, len(in_keys), "in_keys"
        )
        bias = _check_module_map_parameter(bias, "bias", bool, len(in_keys), "in_keys")

        modules: List[Module] = []
        for i in range(len(in_keys)):
            module = torch.nn.Linear(
                in_features=in_features[i],
                out_features=out_features[i],
                bias=bias[i],
                device=device,
                dtype=dtype,
            )
            modules.append(module)
        self.module_map = ModuleMap(in_keys, modules, out_properties)

    def forward(self, tensor: TensorMap) -> TensorMap:
        """
        Apply the transformation to the input tensor map `tensor`.

        :param tensor: :py:class:`TensorMap` with the input tensor to be transformed.
        :return: :py:class:`TensorMap`
        """
        return self.module_map(tensor)


class EquivariantLinear(Module):
    """
    Module similar to :py:class:`torch.nn.Linear` that works with equivariant
    :py:class:`metatensor.torch.TensorMap` objects.

    Applies a linear transformation to each block of a :py:class:`TensorMap` passed to
    its forward method, indexed by :param in_keys:.

    Refer to the :py:class`torch.nn.Linear` documentation for a more detailed
    description of the other parameters.

    Each parameter can be passed as a single value of its expected type, which is used
    as the parameter for all blocks. Alternatively, they can be passed as a list to
    control the parameters applied to each block indexed by the keys in :param in_keys:.

    For :py:class:`EquivariantLinear`, by contrast to :py:class:`Linear`, the parameter
    :param bias: is only applied to modules corresponding to invariant blocks, i.e.
    those indexed by keys in :param in_keys` at numeric indices :param
    invariant_key_idxs:. If passed as a list, :param bias: must therefore have the same
    length as :param invariant_key_idxs:.

    :param in_keys: :py:class:`Labels`, the keys that are assumed to be in the input
        tensor map in the :py:meth:`forward` method.
    :param invariant_key_idxs: list of int, the indices of the invariant keys present in
        `in_keys` in the input :py:class:`TensorMap`. Only blocks for these keys will
        have bias applied according to the user choice. Covariant blocks will not have
        bias applied.
    :param out_properties: list of :py:class`Labels` (optional), the properties labels
        of the output. By default the output properties are relabeled using
        Labels.range. If provided, :param out_features: can be inferred and need not be
        provided.
    """

    def __init__(
        self,
        in_keys: Labels,
        invariant_key_idxs: List[int],
        in_features: Union[int, List[int]],
        out_features: Optional[Union[int, List[int]]] = None,
        out_properties: Optional[List[Labels]] = None,
        *,
        bias: Union[bool, List[bool]] = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # Infer `out_features` if not provided
        if out_features is None:
            if out_properties is None:
                raise ValueError(
                    "If `out_features` is not provided,"
                    " `out_properties` must be provided."
                )
            out_features = [len(p) for p in out_properties]

        # Check input parameters, convert to lists (for each key) if necessary
        in_features = _check_module_map_parameter(
            in_features, "in_features", int, len(in_keys), "in_keys"
        )
        out_features = _check_module_map_parameter(
            out_features, "out_features", int, len(in_keys), "in_keys"
        )
        bias = _check_module_map_parameter(
            bias, "bias", bool, len(invariant_key_idxs), "invariant_key_idxs"
        )

        modules: List[Module] = []
        for i in range(len(in_keys)):
            if (
                i in invariant_key_idxs
            ):  # Invariant block: apply bias according to user choice
                for j in range(len(invariant_key_idxs)):
                    if invariant_key_idxs[j] == i:
                        bias_block = bias[j]
            else:  # Covariant block: do not apply bias
                bias_block = False

            module = torch.nn.Linear(
                in_features=in_features[i],
                out_features=out_features[i],
                bias=bias_block,
                device=device,
                dtype=dtype,
            )
            modules.append(module)

        self.module_map = ModuleMap(in_keys, modules, out_properties)

    def forward(self, tensor: TensorMap) -> TensorMap:
        """
        Apply the transformation to the input tensor map `tensor`.

        :param tensor: :py:class:`TensorMap` with the input tensor to be transformed.
        :return: :py:class:`TensorMap`
        """
        return self.module_map(tensor)
