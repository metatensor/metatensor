"""
.. _learn-tutorial-arbitrary-architectures:

Using neural network modules
============================

.. py:currentmodule:: metatensor.torch.learn.nn

This tutorial demonstrates the use of convenience modules in 
metatensor-learn to build simple multi-layer perceptrons.

"""

import torch

import metatensor.torch as mts
from metatensor.torch import TensorMap, TensorBlock, Labels
from metatensor.torch.learn.nn import ModuleMap

# %%
#
# Defining arbitrary architectures with ``ModuleMap``
# ---------------------------------------------------
#
# The previous section outlines how to use metatensor learn's ``nn`` modules to build
# simple multi-layer perceptrons. Now we will explore the use of a special module called
# ``ModuleMap`` that allows users to wrap any native torch module to be compatible with
# a ``TensorMap``.
#
# This is useful for building arbitrary architectures containing layers more complex
# than found in the standard available layers: namely ``Linear``, ``Tanh``, ``ReLU``,
# ``SiLU`` and ``LayerNorm`` and their equivariant counterparts.
#
# To demonstrate this functionality, it is most instructive to use a set of
# pre-calculated equivariant features. We will use the spherical expansion from the
# :ref:`first steps tutorial <core-tutorial-first-steps>`, and build custom NN
# architectures to preserve invariance for invariant blocks and covariance for covariant
# blocks.

# First, load the spherical expansion 
spherical_expansion = mts.load("../core/spherical-expansion.mts")

# metatensor-learn modules currently do not support TensorMaps with gradients
spherical_expansion = mts.remove_gradients(spherical_expansion)
print(spherical_expansion)

# %% This time we're going to define our target as a global (i.e. per-system) symmetric
# matrix (such as the polarizability tensor), which when expressed in the spherical
# basis is a spherical tensor of angular orders l = [0, 2].
#
# TODO!
# 
# Load polarizability from disk: re-use Paolo's data for the polarizability here?
#
# Build an arbitrary torch.nn.Sequential for each block separately and then wrap in a ModuleMap
#
# Define the same tensormap loss as the 1st NN tutorial and run the training loop