"""
.. _learn-tutorial-nn-modules-basic:

Using neural network modules
============================

.. py:currentmodule:: metatensor.torch.learn.nn

This example demonstrates the use of convenience modules in metatensor-learn to build
simple multi-layer perceptrons.

"""

import torch

import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.learn.nn import Linear, ReLU, Sequential


torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

# %%
#
# A brief overview of native ``torch.nn``
# ---------------------------------------
#
# metatensor-learn's neural network modules are designed as
# :py:class:`TensorMap`-compatible analogues to the torch API.
#
# Let's first briefly cover torch's ``nn`` modules to see how they work.
#
# First, let's define a random tensor that we will treat as some intermediate
# representation. We will build a multi-layer perceptron to transform this tensor into a
# prediction.
#
# Let's say we have 100 samples, the size of the input latent space is 128, and the
# target property is of dimension 1. We will start with a simple linear layer to map the
# latent representation to a prediction of the target

n_samples = 100
in_features = 128
out_features = 1
feature_tensor = torch.randn(n_samples, in_features)

# define a dummy target
target_tensor = torch.randn(n_samples, 1)

# initialize the linear layer
linear_torch = torch.nn.Linear(in_features, out_features, bias=True)

# define a loss function
loss_fn = torch.nn.MSELoss(reduction="sum")


# construct a basic training loop. For simplicity do not use datasets or dataloaders.
def training_loop(model, features, targets, loss_fn):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1001):
        optimizer.zero_grad()

        predictions = model(features)

        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch: {epoch}, loss: {loss}")


print("with NN = [Linear]")
training_loop(linear_torch, feature_tensor, target_tensor, loss_fn)

# now define a nonlinear multi-layer perceptron using ``torch.nn.Sequential``
hidden_layer_width = 64
mlp_torch = torch.nn.Sequential(
    torch.nn.Linear(in_features, hidden_layer_width),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer_width, out_features),
)

# run training again
print("with NN = [Linear, ReLU, Linear]")
training_loop(mlp_torch, feature_tensor, target_tensor, loss_fn)

# %%
#
# Using metatensor-learn ``nn`` layers
# ------------------------------------
#
# Now we're ready to see how the ``nn`` module in ``metatensor-learn`` works.
#
# First we need to create some dummy data, this time in :py:class:`TensorMap` format
#
# Starting simple, we will define a :py:class:`TensorMap` with a single
# :py:class:`TensorBlock` containing the latent space features from above.

feature_tensormap = TensorMap(
    keys=Labels.single(),
    blocks=[
        TensorBlock(
            values=feature_tensor,
            samples=Labels(
                ["sample"],
                torch.arange(n_samples).reshape(-1, 1),
            ),
            components=[],
            properties=Labels(
                ["property"],
                torch.arange(in_features).reshape(-1, 1),
            ),
        ),
    ],
)

target_tensormap = TensorMap(
    keys=Labels.single(),
    blocks=[
        TensorBlock(
            values=target_tensor,
            samples=Labels(
                ["sample"],
                torch.arange(n_samples).reshape(-1, 1),
            ),
            components=[],
            properties=Labels(
                ["target"],
                torch.arange(out_features).reshape(-1, 1),
            ),
        ),
    ],
)

# for supervised learning, inputs and labels must have the same metadata for all axes
# except the properties dimension, as this is the dimension that is transformed by the
# NN.
if mts.equal_metadata(
    feature_tensormap, target_tensormap, check=["samples", "components"]
):
    print("metadata check passed!")
else:
    raise ValueError(
        "input and output TensorMaps must have matching keys, samples, "
        "and components metadata"
    )

# use metatensor-learn's Linear layer. We need to pass the target properties labels so
# that the prediction TensorMap is annotated with the correct metadata.
in_keys = feature_tensormap.keys
linear_mts = Linear(
    in_keys=in_keys,
    in_features=in_features,
    out_properties=[block.properties for block in target_tensormap],
    bias=True,
)


# define a custom loss function for TensorMaps that computes the squared error and
# reduces by sum
def squared_loss(input: TensorMap, target: TensorMap) -> torch.Tensor:
    """
    Computes the total squared error between the ``input`` and ``target`` TensorMaps.
    """
    # input and target should have equal metadata over all axes
    assert mts.equal_metadata(input, target)

    squared_loss = 0
    for key in input.keys:
        squared_loss += torch.sum((input[key].values - target[key].values) ** 2)

    return squared_loss


loss_fn_mts = squared_loss

# run the training loop
print("with NN = [Linear]")
training_loop(linear_mts, feature_tensormap, target_tensormap, loss_fn_mts)


# now construct a nonlinear MLP instead. Here we use metatensor-learn's Sequential
# module, along with some nonlinear activation modules. We only need to pass the
# properties metadata for the output layer, for the hidden layers, we can just pass the
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
training_loop(mlp_mts, feature_tensormap, target_tensormap, loss_fn_mts)
