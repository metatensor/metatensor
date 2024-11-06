"""
Module containing classes :class:`LayerNorm` and :class:`InvariantLayerNorm`, i.e.
module maps that apply layer norms in a generic and equivariant way,
respectively.
"""

from typing import List, Optional

import torch
from torch.nn import Module, init
from torch.nn.parameter import Parameter

from .._backend import Labels, TensorMap
from .._dispatch import int_array_like
from ._utils import _check_module_map_parameter
from .module_map import ModuleMap


class LayerNorm(Module):
    """
    Module similar to :py:class:`torch.nn.LayerNorm` that works with
    :py:class:`metatensor.torch.TensorMap` objects.

    Applies a layer normalization to each block of a :py:class:`TensorMap` passed to its
    :py:meth:`forward` method, indexed by :param in_keys:.

    The main difference from :py:class:`torch.nn.LayerNorm` is that there is no
    `normalized_shape` parameter. Instead, the standard deviation and mean (if
    applicable) are calculated over all dimensions except the samples (first) dimension
    of each :py:class:`TensorBlock`.

    The extra parameter :param mean: controls whether or not the mean over these
    dimensions is subtracted from the input tensor in the transformation.

    Refer to the :py:class`torch.nn.LayerNorm` documentation for a more detailed
    description of the other parameters.

    Each parameter is passed as a single value of its expected type, which is used
    as the parameter for all blocks.

    :param in_keys: :py:class:`Labels`, the keys that are assumed to be in the input
        tensor map in the :py:meth:`forward` method.
    :param in_features: list of int, the number of features in the input tensor for each
        block indexed by the keys in :param in_keys:. If passed as a single value, the
        same number of features is assumed for all blocks.
    :param out_properties: list of :py:class`Labels` (optional), the properties labels
        of the output. By default (if none) the output properties are relabeled using
        Labels.range.
    :mean bool: whether or not to subtract the mean over all dimensions except the
        samples (first) dimension of each block of the input passed to
        :py:meth:`forward`.
    """

    def __init__(
        self,
        in_keys: Labels,
        in_features: List[int],
        out_properties: Optional[List[Labels]] = None,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        mean: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # Check input parameters, convert to lists (for each key) if necessary
        in_features = _check_module_map_parameter(
            in_features, "in_features", int, len(in_keys), "in_keys"
        )
        eps = _check_module_map_parameter(eps, "eps", float, len(in_keys), "in_keys")
        elementwise_affine = _check_module_map_parameter(
            elementwise_affine, "elementwise_affine", bool, len(in_keys), "in_keys"
        )
        bias = _check_module_map_parameter(bias, "bias", bool, len(in_keys), "in_keys")
        mean = _check_module_map_parameter(mean, "mean", bool, len(in_keys), "in_keys")

        # Build module list
        modules: List[Module] = []
        for i in range(len(in_keys)):
            module = _LayerNorm(
                in_features=in_features[i],
                eps=eps[i],
                elementwise_affine=elementwise_affine[i],
                bias=bias[i],
                mean=mean[i],
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
        # Currently not supporting gradients
        if len(tensor[0].gradients_list()) != 0:
            raise ValueError(
                "Gradients not supported. Please use metatensor.remove_gradients()"
                " before using this module"
            )
        return self.module_map(tensor)


class InvariantLayerNorm(Module):
    """
    Module similar to :py:class:`torch.nn.LayerNorm` that works with
    :py:class:`metatensor.torch.TensorMap` objects, applying the transformation only to
    the invariant blocks.

    Applies a layer normalization to each invariant block of a :py:class:`TensorMap`
    passed to :py:meth:`forward` method. These are indexed by
    the keys in :param in_keys: that correspond to the selection passed in :param
    invariant_keys:.

    The main difference from :py:class:`torch.nn.LayerNorm` is that there is no
    `normalized_shape` parameter. Instead, the standard deviation and mean (if
    applicable) are calculated over all dimensions except the samples (first) dimension
    of each :py:class:`TensorBlock`.

    The extra parameter :param mean: controls whether or not the mean over these
    dimensions is subtracted from the input tensor in the transformation.

    Refer to the :py:class`torch.nn.LayerNorm` documentation for a more detailed
    description of the other parameters.

    Each parameter is passed as a single value of its expected type, which is used
    as the parameter for all blocks.

    :param in_keys: :py:class:`Labels`, the keys that are assumed to be in the input
        tensor map in the :py:meth:`forward` method.
    :param in_features: list of int, the number of features in the input tensor for each
        block indexed by the keys in :param in_keys:. If passed as a single value, the
        same number of features is assumed for all blocks.
    :param out_properties: list of :py:class`Labels` (optional), the properties labels
        of the output. By default (if none) the output properties are relabeled using
        Labels.range.
    :param invariant_keys: a :py:class:`Labels` object that is used to select the
        invariant keys from ``in_keys``. If not provided, the invariant keys are assumed
        to be those where key dimensions ``["o3_lambda", "o3_sigma"]`` are equal to
        ``[0, 1]``.
    :mean bool: whether or not to subtract the mean over all dimensions except the
        samples (first) dimension of each block of the input passed to
        :py:meth:`forward`.
    """

    def __init__(
        self,
        in_keys: Labels,
        in_features: List[int],
        out_properties: Optional[List[Labels]] = None,
        invariant_keys: Optional[Labels] = None,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        mean: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        # Set a default for invariant keys
        if invariant_keys is None:
            invariant_keys = Labels(
                names=["o3_lambda", "o3_sigma"],
                values=int_array_like([0, 1], like=in_keys.values).reshape(-1, 2),
            )
        invariant_key_idxs = in_keys.select(invariant_keys)

        # Check input parameters, convert to lists (for each *invariant* key) if
        # necessary.
        in_features = _check_module_map_parameter(
            in_features,
            "in_features",
            int,
            len(invariant_key_idxs),
            "invariant_key_idxs",
        )
        eps = _check_module_map_parameter(
            eps, "eps", float, len(invariant_key_idxs), "invariant_key_idxs"
        )
        elementwise_affine = _check_module_map_parameter(
            elementwise_affine,
            "elementwise_affine",
            bool,
            len(invariant_key_idxs),
            "invariant_key_idxs",
        )
        bias = _check_module_map_parameter(
            bias, "bias", bool, len(invariant_key_idxs), "invariant_key_idxs"
        )
        mean = _check_module_map_parameter(
            mean, "mean", bool, len(invariant_key_idxs), "invariant_key_idxs"
        )

        # Build module list
        modules: List[Module] = []
        invariant_idx: int = 0
        for i in range(len(in_keys)):
            if i in invariant_key_idxs:  # Invariant block: apply LayerNorm
                for j in range(len(invariant_key_idxs)):
                    if invariant_key_idxs[j] == i:
                        invariant_idx = j

                module = _LayerNorm(
                    in_features=in_features[invariant_idx],
                    eps=eps[invariant_idx],
                    elementwise_affine=elementwise_affine[invariant_idx],
                    bias=bias[invariant_idx],
                    mean=mean[invariant_idx],
                    device=device,
                    dtype=dtype,
                )
            else:  # Covariant block: apply the identity operator
                module = torch.nn.Identity()
            modules.append(module)
        self.module_map = ModuleMap(in_keys, modules, out_properties)

    def forward(self, tensor: TensorMap) -> TensorMap:
        """
        Apply the layer norm to the input tensor map `tensor`.

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


class _LayerNorm(Module):
    """
    Custom :py:class:`Module` re-implementing :py:class:`torch.nn.LayerNorm`.

    In this case, `normalized_shape` is not provided as a parameter. Instead, the
    standard deviation and mean (if applicable) are calculated over all dimensions
    except the samples of the input tensor.

    Subtraction of this mean can be switched on or off using the extra :param mean:
    parameter.

    Refer to :py:class:`torch.nn.LayerNorm` documentation for more information on the
    other parameters.

    :param mean: bool, whether or not to subtract the mean over all dimension except the
        samples of the input tensor passed to :py:meth:`forward`.
    """

    __constants__ = ["in_features", "eps", "elementwise_affine"]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        in_features: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        mean: bool = True,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.mean = mean
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(in_features, device=device, dtype=dtype)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(in_features, device=device, dtype=dtype)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _layer_norm(
            input, weight=self.weight, bias=self.bias, eps=self.eps, mean=self.mean
        )

    def extra_repr(self) -> str:
        return "eps={eps}, elementwise_affine={elementwise_affine}".format(
            **self.__dict__
        )


def _layer_norm(
    tensor: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    mean: bool,
) -> torch.Tensor:
    """
    Apply layer normalization to the input `tensor`.

    See :py:class:`torch.nn.functional.layer_norm` for more information on the other
    parameters.

    In addition to base torch implementation, this function has the added control of
    whether or not the mean over all dimensions except the samples (first) dimension is
    subtracted from the input tensor.

    :param mean: whether or not to subtract from the input :param tensor: the mean over
        all dimensions except the samples (first) dimension.

    :return: :py:class:`torch.Tensor` with layer normalization applied.
    """
    # Contract over all dimensions except samples
    dim: List[int] = list(range(1, len(tensor.shape)))

    if mean:  # subtract mean over properties dimension
        tensor_out = tensor - torch.mean(tensor, dim=dim, keepdim=True)
    else:
        tensor_out = tensor

    # Divide by standard deviation over properties dimension. `correction=0` for biased
    # estimator, in accordance with the torch implementation.
    tensor_out /= torch.sqrt(
        torch.var(tensor, dim=dim, correction=0, keepdim=True) + eps
    )

    if weight is not None:  # apply affine transformation
        tensor_out *= weight

    if bias is not None:  # apply bias
        tensor_out += bias

    return tensor_out
