from typing import List, Optional, Union

import torch
from torch.nn import Module

from .._backend import Labels, TensorMap
from .module_map import ModuleMap
from ._utils import _check_module_map_parameter


class EquivariantTransform(Module):
    """
    Applies a module to invariant blocks and to the invariants commputed from each
    covariant block of a :py:class:`TensorMap` passed to its forward method, indexed by
    :param in_keys:. For covariant blocks, the transformed values are then elementwise
    multiplied to the original block, without affecting its `component` dimension.

    Each parameter can be passed as a single value of its expected type, which is used
    as the parameter for all blocks. Alternatively, they can be passed as a list to
    control the parameters applied to each block indexed by the keys in :param in_keys:.

    :param module: :py:class:`callable`, the callable to apply to the invariant blocks
        and to the invariant built from the covariant blocks.
    :param in_keys: :py:class:`Labels`, the keys that are assumed to be in the input
        tensor map in the :py:meth:`forward` method.
    :param invariant_key_idxs: list of int, the indices of the invariant keys present in
        `in_keys` in the input :py:class:`TensorMap`. Only blocks for these keys will
        have bias applied according to the user choice. Covariant blocks will not have
        bias applied.
    :param in_features: list of int, the number of features in the input tensor for each
        block indexed by the keys in :param in_keys:. If passed as a single value, the
        same number of features is assumed for all blocks.
    :param out_properties: list of :py:class`Labels` (optional), the properties labels
        of the output. By default the output properties are relabeled using
        Labels.range. If provided, :param out_features: can be inferred and need not be
        provided.
    :param device: :py:class:`torch.device` Device to instantiate the module in.
    :param dtype: :py:class:`torch.dtype` passed to the module instance.
    """

    def __init__(
        self,
        module: callable,
        in_keys: Labels,
        invariant_key_idxs: List[int],
        in_features: Union[int, List[int]],
        out_features: Optional[Union[int, List[int]]] = None,
        out_properties: Optional[List[Labels]] = None,
        *,
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

        modules: List[Module] = []
        for i in range(len(in_keys)):
            if i in invariant_key_idxs:
                module_i = module
            else:
                module_i = _CovariantTransform(
                    in_features=in_features[i],
                    module=module,
                    device=device,
                    dtype=dtype,
                )
            modules.append(module_i)

        self.module_map = ModuleMap(in_keys, modules, out_properties)

    def forward(self, tensor: TensorMap) -> TensorMap:
        """
        Apply the transformation to the input tensor map `tensor`.

        :param tensor: :py:class:`TensorMap` with the input tensor to be transformed.
        :return: :py:class:`TensorMap`
        """
        return self.module_map(tensor)


class _CovariantTransform(Module):
    """
    Custom :py:class:`Module` to apply a generic :py:class:`callable` to invariants
    built from a covariant block.

    :param in_features: a :py:class:`int`, the input feature dimension. This also corresponds to the output
        feature size as the shape of the tensor passed to :py:meth:`forward` is preserved.
    :param module: a :py:class:`callable` that when called with parameters ``in_features`` and optionally ``dtype``
        and ``device`` constructs a native :py:class:`torch.nn.Module`.
    :param device: :py:class:`torch.device`
        Device to instantiate the modules in.
    :param dtype: :py:class:`torch.dtype`
        dtype of the module.
    """

    def __init__(
        self,
        in_features: int,
        module: callable,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:

        super().__init__()

        self.in_features = in_features
        self.module = module(in_features, device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
    """
    Creates an invariant block from the ``input`` covariant, and transforms it by applying the torch 
    ``module`` passed to the class constructor. Then uses the transformed invariant as an element
    -wise multiplier for the ``input`` block. 
    
    Transformations are applied consistently to components (axis 1) to preserve equivariance.
    """
        assert len(input.shape) == 3, "``input`` must be a three-dimensional tensor"
        invariant = input.norm(dim=1, keepdim=True)
        invariant_transformed = self.module(invariant_components)
        tensor_out = invariant_transformed * input

        return tensor_out
