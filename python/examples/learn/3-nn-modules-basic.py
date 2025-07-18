"""
.. _learn-tutorial-nn-modules-basic:

Convenience ``nn`` modules
==========================

.. py:currentmodule:: metatensor.torch.learn.nn

This example demonstrates the use of convenience modules in metatensor-learn to build
simple multi-layer perceptrons.

.. note::

    The convenience modules introduced in this tutorial are designed to be used to
    prototype new architectures for simple models. If you already have more complex
    models, you can also wrap them in ``ModuleMap`` objects to make them compatible with
    metatensor. This is covered in a later tutorial :ref:`using module maps
    <learn-tutorial-nn-using-modulemap>`.
"""

from typing import Union

import torch

import metatensor.torch as mts
from metatensor.torch import Labels, TensorMap
from metatensor.torch.learn.nn import Linear, ReLU, Sequential


torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

# %%
#
# Introduction to native ``torch.nn`` modules
# -------------------------------------------
#
# metatensor-learn's neural network modules are designed to be
# :py:class:`TensorMap`-compatible analogues to the torch API. Before looking into the
# ``metatensor-learn version``, it is instructive to recap torch's native ``nn`` modules
# and recall how they work.
#
# First, let's define a random tensor that we will treat as an intermediate
# representation. We will build a multi-layer perceptron to transform this tensor into a
# prediction.
#
# Let's say we have 100 samples, the size of the input latent space is 128, and the
# target property is of dimension 1. We will start with a simple linear layer to map the
# latent representation to a prediction of the target property:

n_samples = 100
in_features = 128
out_features = 1
feature_tensor = torch.randn(n_samples, in_features)

# define a dummy target
target_tensor = torch.randn(n_samples, 1)

# initialize the torch linear layer
linear_torch = torch.nn.Linear(in_features, out_features, bias=True)

# define a loss function
loss_fn_torch = torch.nn.MSELoss(reduction="sum")


# construct a basic training loop. For brevity we will not use datasets or dataloaders.
def training_loop(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    features: Union[torch.Tensor, TensorMap],
    targets: Union[torch.Tensor, TensorMap],
) -> None:
    """A basic training loop for a model and it's loss function."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1001):
        optimizer.zero_grad()

        predictions = model(features)

        if isinstance(predictions, torch.ScriptObject):
            # assume a TensorMap and check that the
            # metadata is equivalent
            assert mts.equal_metadata(predictions, targets)

        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch: {epoch}, loss: {loss}")


print("with NN = [Linear]")
training_loop(linear_torch, loss_fn_torch, feature_tensor, target_tensor)

# %%
#
# Now run the training loop, this time with a nonlinear multi-layer perceptron using
# ``torch.nn.Sequential``
hidden_layer_width = 64
mlp_torch = torch.nn.Sequential(
    torch.nn.Linear(in_features, hidden_layer_width),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer_width, out_features),
)

# train again
print("with NN = [Linear, ReLU, Linear]")
training_loop(mlp_torch, loss_fn_torch, feature_tensor, target_tensor)

# %%
#
# Using the metatensor-learn ``nn`` layers
# ----------------------------------------
#
# Now we're ready to see how the ``nn`` module in ``metatensor-learn`` works.
#
# First we create some dummy data, this time in :py:class:`TensorMap` format.
# Starting simple, we will define a :py:class:`TensorMap` with only one
# :py:class:`TensorBlock`, containing the latent space features from above.

feature_tensormap = TensorMap(
    keys=Labels.single(),
    blocks=[mts.block_from_array(feature_tensor)],
)

target_tensormap = TensorMap(
    keys=Labels.single(),
    blocks=[mts.block_from_array(target_tensor)],
)

# for supervised learning, inputs and labels must have the same metadata for all axes
# except the properties dimension, as this is the dimension that is transformed by the
# neural network.
if mts.equal_metadata(
    feature_tensormap, target_tensormap, check=["samples", "components"]
):
    print("metadata check passed!")
else:
    raise ValueError(
        "input and output TensorMaps must have matching keys, samples, "
        "and components metadata"
    )

# use metatensor-learn's Linear layer. We need to pass the target property's labels so
# that the TensorMap for predictions is annotated with the correct metadata.
in_keys = feature_tensormap.keys
linear_mts = Linear(
    in_keys=in_keys,
    in_features=in_features,
    out_properties=[block.properties for block in target_tensormap],
    bias=True,
)


# define a custom loss function over TensorMaps that computes the squared error and
# reduces by summation
class TensorMapLoss(torch.nn.Module):
    """
    A custom loss function for TensorMaps that computes the squared error and
    reduces by summation.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: TensorMap, target: TensorMap) -> torch.Tensor:
        """
        Computes the total squared error between the ``input`` and ``target``
        TensorMaps.
        """
        # input and target should have the same metadata across all axes
        assert mts.equal_metadata(input, target)

        squared_loss = 0
        for key in input.keys:
            squared_loss += torch.sum((input[key].values - target[key].values) ** 2)

        return squared_loss


loss_fn_mts = TensorMapLoss()

# run the training loop
print("with NN = [Linear]")
training_loop(linear_mts, loss_fn_mts, feature_tensormap, target_tensormap)

# %%
#
# Now construct a nonlinear MLP instead. Here we use metatensor-learn's Sequential
# module, along with some nonlinear activation modules. We only need to pass the
# properties metadata for the output layer, for the hidden layers, we just pass the
# layer dimension
mlp_mts = Sequential(
    in_keys,
    Linear(
        in_keys=in_keys,
        in_features=in_features,
        out_features=hidden_layer_width,
        bias=True,
    ),
    ReLU(in_keys=in_keys),  # can also use Tanh or SiLU
    Linear(
        in_keys=in_keys,
        in_features=hidden_layer_width,
        out_properties=[block.properties for block in target_tensormap],
        bias=True,
    ),
)


# run the training loop
print("with NN = [Linear, ReLU, Linear]")
training_loop(mlp_mts, loss_fn_mts, feature_tensormap, target_tensormap)


# %%
#
# Conclusion
# ----------
#
# This tutorial introduced the convenience modules in metatensor-learn for building
# simple neural networks. As we've seen, the API is similar to native ``torch.nn`` and
# the ``TensorMap`` data type can be easily switched in place for torch Tensors in
# existing training loops with minimal changes.
#
# Combined with other learning utilities to construct Datasets and Dataloaders, covered
# in :ref:`basic <learn-tutorial-dataset-dataloader>` and :ref:`advanced
# <learn-tutorial-indexed-dataset-dataloader>` tutorials, metatensor-learn provides a
# powerful framework for building and training machine learning models based on the
# TensorMap data format.
