from typing import List, Optional, Union

import torch
from torch.nn import Module

from .._backend import Labels, TensorMap
from .._dispatch import int_array_like
from ._utils import _check_module_map_parameter
from .module_map import ModuleMap


class EquivariantTransform(Module):
    """
    A custom :py:class:`torch.nn.Module` that applies an arbitrary shape- and
    equivariance-preserving transformation to an input :py:class:`TensorMap`.

    ``module`` is passed as a callable with parameters ``in_features`` and optionally
    ``dtype`` and ``device``. This callable constructs an arbitrary shape-preserving
    :py:class:`torch.nn.Module` transformation (i.e. :py:class:`torch.nn.Tanh`).
    Separate instantiations are created for each block using the metadata information
    passed in ``in_keys`` and ``in_features``.

    For invariant blocks in `in_keys` and indexed by `invariant_key_idxs`, the
    transformation is applied as is. For covariant blocks, an invariant multiplier that
    preserves covariance is created from the transformation.

    Each parameter can be passed as a single value of its expected type, which is used
    as the parameter for all blocks. Alternatively, they can be passed as a list to
    control the parameters applied to each block indexed by the keys in :param in_keys:.

    :param module: a :py:class:`list` of :py:class:`torch.nn.Module` containing the
        transformations to be applied to each block indexed by
        :param:`in_keys`. Transformations for invariant and covariant blocks differ. See
        above.
    :param in_keys: :py:class:`Labels`, the keys that are assumed to be in the input
        :py:class:`TensorMap` in the :py:meth:`forward` method.
    :param in_features: list of int, the number of features in the input tensor for each
        block indexed by the keys in :param in_keys:. If passed as a single value, the
        same number of features is assumed for all blocks.
    :param out_properties: list of :py:class`Labels` (optional), the properties labels
        of the output. By default the output properties are relabeled using
        Labels.range.
    :param invariant_keys: a :py:class:`Labels` object that is used to select the
        invariant keys from ``in_keys``. If not provided, the invariant keys are assumed
        to be those where key dimensions ``["o3_lambda", "o3_sigma"]`` are indexed by
        ``[0, 1]``.
    """

    def __init__(
        self,
        modules: List[torch.nn.Module],
        in_keys: Labels,
        in_features: Union[int, List[int]],
        out_features: Optional[Union[int, List[int]]] = None,
        out_properties: Optional[List[Labels]] = None,
        invariant_keys: Optional[Labels] = None,
    ) -> None:
        super().__init__()

        # Set a default for invariant keys
        if invariant_keys is None:
            invariant_keys = Labels(
                names=["o3_lambda", "o3_sigma"],
                values=int_array_like([0, 1], like=in_keys.values).reshape(-1, 1),
            )
        invariant_key_idxs = in_keys.select(invariant_keys)

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

        modules_for_map: List[Module] = []
        for i in range(len(in_keys)):
            if i in invariant_key_idxs:
                module_i = modules[i]
            else:
                module_i = _CovariantTransform(
                    module=modules[i],
                )
            modules_for_map.append(module_i)

        self.module_map = ModuleMap(in_keys, modules_for_map, out_properties)

    def forward(self, tensor: TensorMap) -> TensorMap:
        """
        Apply the transformation to the input tensor map `tensor`.

        :param tensor: :py:class:`TensorMap` with the input tensor to be transformed.
        :return: :py:class:`TensorMap` corresponding to the transformed input ``tensor``.
        """
        return self.module_map(tensor)


class _CovariantTransform(Module):
    """
    Applies an arbitrary shape-preserving transformation defined in ``module`` to a
    3-dimensional tensor in a way that preserves equivariance.

    :param in_features: a :py:class:`int`, the input feature dimension. This also
        corresponds to the output feature size as the shape of the tensor passed to
        :py:meth:`forward` is preserved.
    :param module: :py:class:`torch.nn.Module` containing the transformation to be
        applied to the invariants constructed from the norms over the component
        dimension of the input :py:class:`torch.Tensor` passed to the :py:meth:`forward`
        method.
    """

    def __init__(
        self,
        module: torch.nn.Module,
    ) -> None:
        super().__init__()

        self.module = module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Creates an invariant block from the ``input`` covariant, and transforms it by
        applying the torch ``module`` passed to the class constructor. Then uses the
        transformed invariant as an elementwise multiplier for the ``input`` block.

        Transformations are applied consistently to components (axis 1) to preserve
        equivariance.
        """
        assert len(input.shape) == 3, "``input`` must be a three-dimensional tensor"
        invariant = input.norm(dim=1, keepdim=True)
        invariant_transformed = self.module(invariant)
        tensor_out = invariant_transformed * input

        return tensor_out
