"""
.. _learn-tutorial-nn-modules-equivariant:

Using neural network modules
============================

.. py:currentmodule:: metatensor.torch.learn.nn

This example demonstrates the use of convenience modules in metatensor-learn to build
simple equivariance-preserving multi-layer perceptrons.

"""

import torch

import metatensor.torch as mts
# from metatensor.torch import TensorMap, TensorBlock, Labels
# from metatensor.torch.learn.nn import EquivariantLinear, InvariantSiLU, Sequential

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

# %%
#
# Equivariant architectures
# -------------------------
#
# Many times targets of machine learning can be physical observables with certain
# symmetries. Many successful approaches to these learning tasks use
# equivariance-preserving architectures to map equivariant features onto predictions of
# an equivariant target.
#
#
#
#
# Let's load the spherical expansion from the :ref:`first steps tutorial
# <core-tutorial-first-steps>`: 
spherical_expansion = mts.load("../core/spherical-expansion.mts")

# metatensor-learn modules currently do not support TensorMaps with gradients
spherical_expansion = mts.remove_gradients(spherical_expansion)
print(spherical_expansion)

# TODO!
#
# use EquivariantLinear first and run training loop
#
# then construct a basic equivariant-MLP with Sequential and run training loop
#
