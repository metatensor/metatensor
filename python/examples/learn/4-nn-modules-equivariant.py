"""
.. _learn-tutorial-nn-modules-equivariant:

Equivariance-preserving ``nn`` modules
======================================

.. py:currentmodule:: metatensor.torch.learn.nn

This example demonstrates the use of convenience modules in metatensor-learn to build
simple equivariance-preserving multi-layer perceptrons.

.. note::

    Prior to this tutorial, it is recommended to read the tutorial on :ref:`using
    convenience modules <learn-tutorial-nn-modules-basic>`, as this tutorial builds on
    the concepts introduced there.
"""

import torch

import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.learn.nn import (
    EquivariantLinear,
    InvariantReLU,
    Sequential,
)


torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

# %%
#
# Introduction
# ------------
#
# Often the targets of machine learning are physical observables with certain
# symmetries, such as invariance with respect to translation or equivariance with
# respect to rotation (i.e., rotating the input structure means that the target
# should be rotated in the same way).
#
# Many successful approaches to these learning tasks use equivariance-preserving
# architectures to map equivariant features onto predictions of an equivariant target.
#
# In this example we will demonstrate how to build an equivariance-preserving
# multi-layer perceptron (MLP) on top of some equivariant features.
#
# Let's load the spherical expansion from the :ref:`first steps tutorial
# <core-tutorial-first-steps>`.
spherical_expansion = mts.load("../core/spherical-expansion.mts")

# metatensor-learn modules currently do not support TensorMaps with gradients
spherical_expansion = mts.remove_gradients(spherical_expansion)
print(spherical_expansion)
print("\nNumber of blocks in the spherical expansion:", len(spherical_expansion))

# %%
#
# As a reminder, these are the coefficients of the spherical-basis decompositions of a
# smooth Gaussian density representation of 3D point cloud. In this case, the point
# cloud is a set of decorated atomic positions.
#
# The important part here is that these features are block sparse in angular momentum
# channel (key dimension ``"o3_lambda"``), with each block having a different behaviour
# under rigid rotation by the SO(3) group.
#
# In general, blocks that are invariant under rotation (where ``o3_lambda == 0``) can be
# transformed in arbitrary (i.e. nonlinear) ways in the mapping from features to target,
# while covariant blocks (where ``o3_lambda > 0``) must be transformed in a way that
# preserves the equivariance of the features. The simplest way to do this is to use only
# linear transformations for the latter.


# %%
#
# Define equivariant target data
# ------------------------------
#
# Let's build some dummy target data: we will predict a global (i.e. per-system) rank-2
# symmetric tensor, which decomposes into ``o3_lambda = [0, 2]`` angular momenta
# channels when expressed in the spherical basis. An example of such a target in
# atomistic machine learning is the electronic polarizability of a molecule.
#
# Our target will be block sparse with ``"o3_lambda"`` key dimensions equal to [0, 2],
# and as this is a real- (not pseudo-) tensor, the inversion sigma (``"o3_sigma"``) will
# be +1.
target_tensormap = TensorMap(
    keys=Labels(["o3_lambda", "o3_sigma"], torch.tensor([[0, 1], [2, 1]])),
    blocks=[
        TensorBlock(
            values=torch.randn((1, 1, 1), dtype=torch.float64),
            # only one system
            samples=Labels(["system"], torch.tensor([[0]])),
            # o3_mu = [0]
            components=[Labels(["o3_mu"], torch.tensor([[0]]))],
            # only one 'property' (the L=0 part of the polarizability)
            properties=Labels(["_"], torch.tensor([[0]])),
        ),
        TensorBlock(
            values=torch.randn((1, 5, 1), dtype=torch.float64),
            # only one system
            samples=Labels(["system"], torch.tensor([[0]])),
            # o3_mu = [-2, -1, 0, +1, +2]
            components=[Labels(["o3_mu"], torch.tensor([[-2], [-1], [0], [1], [2]]))],
            # only one 'property' (the L=2 part of the polarizability)
            properties=Labels(["_"], torch.tensor([[0]])),
        ),
    ],
)
print(target_tensormap, target_tensormap[0])

# %%
#
# Filter the feature blocks to only keep the blocks with symmetries that match the
# target: as our target only contains ``o3_lambda = [0, 2]`` channels, we only need
# these!
spherical_expansion = mts.filter_blocks(spherical_expansion, target_tensormap.keys)
print(spherical_expansion)
print(
    "\nNumber of blocks in the filtered spherical expansion:", len(spherical_expansion)
)

# %%
#
# Using equivariant convenience layers
# ------------------------------------
#
# Now we can build our neural network. Our architecture will consist of separate "block
# models", i.e. transformations with separate learnable weights for each block in the
# spherical expansion. This is in contrast to the previous tutorial :ref:`nn modules
# basic <learn-tutorial-nn-modules-basic>`, where we only had a single block in our
# features and targets.
#
# Furthermore, as the features are a per-atom quantity, we will use sparse tensor
# operations to sum over the contributions of all atoms in the system to get a per-sytem
# prediction. For this we will use ``metatensor-operations``.
#
# Starting simple, let's define the neural network as just a simple linear layer. As
# stated before, only linear transformations must be applied to covariant blocks, in
# this case those with ``o3_lambda = 2``, while nonlinear transformations can be applied
# to invariant blocks where ``o3_lambda = 0``. We will use the
# :py:class:`~metatensor.torch.learn.nn.EquivariantLinear` module for this.
in_keys = spherical_expansion.keys
equi_linear = EquivariantLinear(
    in_keys=in_keys,
    in_features=[len(spherical_expansion[key].properties) for key in in_keys],
    out_features=1,  # for all blocks
)
print(in_keys)
print(equi_linear)

# %%
#
# We can see by printing the architecture of the ``EquivariantLinear`` module,
# that there are 8 'Linear' layers, one for each block. In order to preserve
# equivariance, bias is always turned off for all covariant blocks. For
# invariant blocks, bias can be switched on or off by passing the boolean
# parameter ``bias`` when initializing ``EquivariantLinear`` objects.

# Let's see what happens when we pass features through the network.
per_atom_predictions = equi_linear(spherical_expansion)
print(per_atom_predictions)
print(per_atom_predictions[0])

# %%
#
# The outputs of the ``EquivariantLinear`` module are still per-atom and block sparse in
# both "center_type" and "neighbor_type". To get per-system predictions, we can
# "densify" the predictions in these key dimensions by moving them to samples,
# then taking the sum over all sample dimensions except "system".
per_atom_predictions = per_atom_predictions.keys_to_samples(
    ["center_type", "neighbor_type"]
)
per_system_predictions = mts.sum_over_samples(
    per_atom_predictions, ["atom", "center_type", "neighbor_type"]
)
assert mts.equal_metadata(per_system_predictions, target_tensormap)
print(per_system_predictions, per_system_predictions[0])

# %%
#
# The overall 'model' that maps features to targets contains both the application of a
# neural network and some extra transformations, we can wrap it all in a single torch
# module.


class EquivariantMLP(torch.nn.Module):
    """
    A simple equivariant MLP that maps per-atom features to per-structure targets.
    """

    def __init__(self, mlp: torch.nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, features: TensorMap) -> TensorMap:
        # apply the multi-layer perceptron to the features
        per_atom_predictions = self.mlp(features)

        # densify the predictions in the "center_type" and "neighbor_type" key
        # dimensions
        per_atom_predictions = per_atom_predictions.keys_to_samples(
            ["center_type", "neighbor_type"]
        )

        # sum over all sample dimensions except "system"
        per_system_predictions = mts.sum_over_samples(
            per_atom_predictions, ["atom", "center_type", "neighbor_type"]
        )

        return per_system_predictions


# %%
#
# Now we will construct the loss function and run the training loop as we did in the
# previous tutorial, :ref:`nn modules basic <learn-tutorial-nn-modules-basic>`.


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
    features: TensorMap,
    targets: TensorMap,
) -> None:
    """A basic training loop for a model and loss function."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(301):
        optimizer.zero_grad()

        predictions = model(features)

        assert mts.equal_metadata(predictions, targets)

        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch: {epoch}, loss: {loss}")


loss_fn_mts = TensorMapLoss()
model = EquivariantMLP(equi_linear)
print("with NN = [EquivariantLinear]")
training_loop(model, loss_fn_mts, spherical_expansion, target_tensormap)

# %%
#
# Let's inspect the per-block losses using predictions from the trained model. Note that
# the model is able to perfectly fit the invariant target blocks, but not the covariant
# blocks. This is to be expected, as the target data was generated with random numbers
# and is not itself equivariant, making the learning task impossible.
#
# See also the atomistic cookbook example on rotational equivariance for a more detailed
# discussion of this topic:
# https://atomistic-cookbook.org/examples/rotate-equivariants/rotate-equivariants.html
print("per-block loss:")
prediction = model(spherical_expansion)
for key, block in prediction.items():
    print(key, torch.sum((block.values - target_tensormap[key].values) ** 2).item())


# %%
#
# Now let's consider a more complex nonlinear architecture. In the simplest case we are
# restricted to linear layers for covariant blocks, but we can use nonlinear layers for
# invariant blocks.
#
# We will use the :py:class:`~metatensor.torch.learn.nn.InvariantReLU` activation
# function. It has the prefix "Invariant" as it only applies the activation function to
# invariant blocks where ``o3_lambda = 0``, and leaves the covariant blocks unchanged.

# Let's build a new MLP with two linear layers and one activation function.
hidden_layer_width = 64
equi_mlp = Sequential(
    in_keys,
    EquivariantLinear(
        in_keys=in_keys,
        in_features=[len(spherical_expansion[key].properties) for key in in_keys],
        out_features=hidden_layer_width,
    ),
    InvariantReLU(in_keys),  # could also use InvariantTanh, InvariantSiLU
    EquivariantLinear(
        in_keys=in_keys,
        in_features=[hidden_layer_width for _ in in_keys],
        out_features=1,  # for all blocks
    ),
)
print(in_keys)
print(equi_mlp)

# %%
#
# Notice that for invariant blocks, the 'block model' is a nonlinear MLP whereas for
# invariant blocks it is the sequential application of two linear layers, without bias.
# Re-running the training loop with this new architecture:
model = EquivariantMLP(equi_mlp)
print("with NN = [EquivariantLinear, InvariantSiLU, EquivariantLinear]")
training_loop(model, loss_fn_mts, spherical_expansion, target_tensormap)

# %%
#
# With the trained model, let's see the per-block decomposition of the loss. As before,
# the model can perfectly fit the invariants, but not the covariants, as expected.
print("per-block loss:")
prediction = model(spherical_expansion)
for key, block in prediction.items():
    print(key, torch.sum((block.values - target_tensormap[key].values) ** 2).item())

# %%
#
# Conclusion
# ----------
#
# This tutorial has demonstrated how to build equivariance-preserving architectures
# using the metatensor-learn convenience neural network modules. These modules, such as
# ``EquivariantLinear`` and ``InvariantReLU`` are modified analogs of the standard
# convenience layers, such as ``Linear`` and ``ReLU``.
#
# The key difference is that the invariant or covariant nature (via the "o3_lambda" key
# dimension) of the input blocks are taken into account, and used to determine the
# transformations applied to each block.

# %%
#
# Other examples
# --------------
#
# See the atomistic cookbook for an example on learning the polarizability using
# ``EquivariantLinear`` applied to higher body order features:
#
# https://atomistic-cookbook.org/examples/polarizability/polarizability.html
#
# and those for checking the rotational equivariance of quantities in ``TensorMap``
# format:
#
# https://atomistic-cookbook.org/examples/rotate-equivariants/rotate-equivariants.html
