"""
.. _learn-tutorial-nn-using-modulemap:

Custom architectures with ``ModuleMap``
=======================================

.. py:currentmodule:: metatensor.torch.learn.nn

This tutorial demonstrates how to build custom architectures compatible with
``TensorMap`` objects by combining native ``torch.nn`` modules with metatensor-learn's
``ModuleMap``.

.. note::

    Prior to this tutorial, it is recommended to read the tutorial on :ref:`using
    convenience modules <learn-tutorial-nn-modules-basic>`, as this tutorial builds on
    the concepts introduced there.
"""

from typing import List, Union

import torch

import metatensor.torch as mts
from metatensor.torch import Labels, TensorMap
from metatensor.torch.learn.nn import Linear, ModuleMap


torch.manual_seed(42)
torch.set_default_dtype(torch.float64)


# %%
#
# Introduction
# ------------
#
# The previous tutorials cover how to use metatensor learn's ``nn`` convenience modules
# to build simple multi-layer perceptrons and their equivariance-preserving analogs. Now
# we will explore the use of a special module called ``ModuleMap`` that allows users to
# wrap any native torch module in a ``TensorMap`` compatible manner.
#
# This is useful for building arbitrary architectures containing layers more
# complex than those found in the standard available layers: namely ``Linear``,
# ``Tanh``, ``ReLU``, ``SiLU`` and ``LayerNorm`` and their equivariant
# counterparts.
#
# First we need to create some dummy data in the :py:class:`TensorMap` format,
# with multiple :py:class:`TensorBlock` objects. Here we will focus on
# unconstrained architectures, as opposed to equivariance preserving ones. The
# principles in the latter case will be similar, as long as care is taken to
# build architectures with equivarince-preserving transformations.
#
# Let's start by defining a random tensor that we will treat as some
# intermediate representation. We will build a multi-layer perceptron to
# transform this tensor into a prediction. Here we will define a 3-block tensor
# map, with variables with the in and out dimensions for each block.
n_samples = 100
in_features = [64, 128, 256]
out_features = [1, 2, 3]

feature_tensormap = TensorMap(
    keys=Labels(["key"], torch.arange(len(out_features)).reshape(-1, 1)),
    blocks=[
        mts.block_from_array(torch.randn(n_samples, in_feats))
        for in_feats in in_features
    ],
)

target_tensormap = TensorMap(
    keys=Labels(["key"], torch.arange(len(out_features)).reshape(-1, 1)),
    blocks=[
        mts.block_from_array(torch.randn(n_samples, out_feats))
        for out_feats in out_features
    ],
)
print("features:", feature_tensormap)
print("target:", target_tensormap)

# %%
#
# Starting simple
# ---------------
#
# Let's start with a simple linear layer, but this time constructed manually using
# ``ModuleMap``. Here we want a linear layer for each block, with the correct in and out
# feature shapes. The result will be a module that is equivalent to the
# ``metatensor.torch.learn.nn.Linear`` module.

in_keys = feature_tensormap.keys

modules = []
for key in in_keys:
    module = torch.nn.Linear(
        in_features=len(feature_tensormap[key].properties),
        out_features=len(target_tensormap[key].properties),
        bias=True,
    )
    modules.append(module)

# initialize the ModuleMap with the input keys, list of modules, and the output
# property labels' metadata.
linear_mmap = ModuleMap(
    in_keys,
    modules,
    out_properties=[target_tensormap[key].properties for key in in_keys],
)
print(linear_mmap)

# %%
#
# ``ModuleMap`` automatically handles the forward pass for each block indexed by
# the ``in_keys`` used to initialize it. In cases where the input contains more
# keys/blocks than what is present in the ``in_keys` field, the forward pass
# will only be applied to the blocks that are present in the input. The output
# will be a new ``TensorMap`` with the same keys as the input, now with the
# correct output metadata.

# apply the ModuleMap to the whole tensor map of features
prediction_full = linear_mmap(feature_tensormap)

# filter the features to only contain one of the blocks,
# and pass it through the ModuleMap
prediction_subset = linear_mmap(
    mts.filter_blocks(
        feature_tensormap, Labels(["key"], torch.tensor([1]).reshape(-1, 1))
    )
)

print(prediction_full.keys, prediction_full.blocks())
print(prediction_subset.keys, prediction_subset.blocks())


# %%
#
# Now we define a loss function and run a training loop. This is the same as in
# the previous tutorials.


# define a custom loss function for TensorMaps that computes the squared error and
# reduces by a summation operation
class TensorMapLoss(torch.nn.Module):
    """
    A custom loss function for TensorMaps that computes the squared error and reduces by
    sum.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, _input: TensorMap, target: TensorMap) -> torch.Tensor:
        """
        Computes the total squared error between the ``_input`` and ``target``
        TensorMaps.
        """
        # inputs and targets should have the same metadata over all axes
        assert mts.equal_metadata(_input, target)

        squared_loss = 0
        for key in _input.keys:
            squared_loss += torch.sum((_input[key].values - target[key].values) ** 2)

        return squared_loss


# construct a basic training loop. For brevity we will not use datasets or dataloaders.
def training_loop(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    features: Union[torch.Tensor, TensorMap],
    targets: Union[torch.Tensor, TensorMap],
) -> None:
    """A basic training loop for a model and loss function."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(501):
        optimizer.zero_grad()

        predictions = model(features)

        if isinstance(predictions, torch.ScriptObject):
            # assume a TensorMap and check metadata is equivalent
            assert mts.equal_metadata(predictions, targets)

        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch: {epoch}, loss: {loss}")


loss_fn_mts = TensorMapLoss()

print("with NN = [Linear]")
training_loop(linear_mmap, loss_fn_mts, feature_tensormap, target_tensormap)


# %%
#
# More complex architectures
# --------------------------

# Defining more complicated architectures is a matter of building
# ``torch.nn.Sequential`` objects for each block, and wrapping them into a single
# ModuleMap.

hidden_layer_width = 32

modules = []
for key in in_keys:
    module = torch.nn.Sequential(
        torch.nn.LayerNorm(len(feature_tensormap[key].properties)),
        torch.nn.Linear(
            in_features=len(feature_tensormap[key].properties),
            out_features=hidden_layer_width,
            bias=True,
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(
            in_features=hidden_layer_width,
            out_features=len(target_tensormap[key].properties),
            bias=True,
        ),
        torch.nn.Tanh(),
    )
    modules.append(module)

# initialize the ModuleMap as in the previous section.
custom_mmap = ModuleMap(
    in_keys,
    modules,
    out_properties=[target_tensormap[key].properties for key in in_keys],
)
print(custom_mmap)

print("with NN = [LayerNorm, Linear, ReLU, Linear, Tanh]")
training_loop(custom_mmap, loss_fn_mts, feature_tensormap, target_tensormap)


# %%
#
# ModuleMap objects can also be wrapped in a ``torch.nn.torch.nn.Module`` to
# allow construction of complex architectures. For instance, we can have a
# "ResNet"-style neural network module that takes a ModuleMap and applies it,
# then sums with some residual connections. Wikipedia has a good summary and
# diagram of this architectural motif, see:
# https://en.wikipedia.org/wiki/Residual_neural_network .
#
# To do the latter step, we can combine application of the ``ModuleMap`` with a
# ``Linear`` convenience layer from metatensor-learn, and the sparse addition operation
# from ``metatensor-operations`` to build a complex architecture.


class ResidualNetwork(torch.nn.Module):
    def __init__(
        self,
        in_keys: Labels,
        in_features: List[int],
        out_properties: List[Labels],
    ) -> None:
        super().__init__()

        # Build the module map as before
        hidden_layer_width = 32
        modules = []
        for in_feats, out_props in zip(in_features, out_properties, strict=True):
            module = torch.nn.Sequential(
                torch.nn.LayerNorm(in_feats),
                torch.nn.Linear(
                    in_features=in_feats,
                    out_features=hidden_layer_width,
                    bias=True,
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=hidden_layer_width,
                    out_features=len(out_props),
                    bias=True,
                ),
                torch.nn.Tanh(),
            )
            modules.append(module)

        self.module_map = ModuleMap(
            in_keys,
            modules,
            out_properties=out_properties,
        )

        # build the input projection layer
        self.projection = Linear(
            in_keys=in_keys,
            in_features=in_features,
            out_properties=out_properties,
            bias=True,
        )

    def forward(self, features: TensorMap) -> TensorMap:
        # apply the module map to the features
        prediction = self.module_map(features)

        # apply the projection layer to the features
        residual = self.projection(features)

        # add the prediction and residual together using the sparse addition
        # from metatensor-operations
        return mts.add(prediction, residual)


model = ResidualNetwork(
    in_keys=in_keys,
    in_features=in_features,
    out_properties=[block.properties for block in target_tensormap],
)
print("with NN = [LayerNorm, Linear, ReLU, Linear, Tanh] plus residual connections")
training_loop(model, loss_fn_mts, feature_tensormap, target_tensormap)

# %%
#
# Conclusion
# ----------
#
# In this tutorial we have seen how to build custom architectures using ``ModuleMap``.
# This allows for arbitrary architectures to be built, as long as the metadata is
# preserved. We have also seen how to build a custom module that wraps a ``ModuleMap``
# and adds residual connections.
#
# The key takeaway is that ``ModuleMap`` can be used to wrap any combination of native
# ``torch.nn`` modules to make them compatible with ``TensorMap``. In combination with
# convenience layers seen in the tutorial :ref:`nn modules basic
# <learn-tutorial-nn-modules-basic>`, and sparse-data operations from
# ``metatensor-operations``, complex architectures can be built with ease.
