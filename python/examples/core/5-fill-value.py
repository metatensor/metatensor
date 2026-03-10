r"""
.. _core-tutorial-fill-value:

Controlling missing data with ``fill_value``
============================================

When merging blocks with :py:meth:`TensorMap.keys_to_properties` or
:py:meth:`TensorMap.keys_to_samples`, some entries in the merged array may not
exist in the original blocks. By default these entries are filled with zero, but
the ``fill_value`` parameter lets you choose a different value -- such as NaN --
to distinguish genuine zeros from missing data.

This tutorial produces a merged TensorMap where missing entries are NaN, then
shows how to build a mask that identifies them.

.. py:currentmodule:: metatensor
"""

# %%
#
# Setup
# -----
#
# We build a small TensorMap with two blocks that have partially overlapping
# samples. Block 0 (species=1) has samples for atoms 0 and 2, while block 1
# (species=6) has samples for atoms 1 and 2. Atom 2 appears in both blocks;
# atoms 0 and 1 each appear in only one.

import numpy as np

import metatensor

keys = metatensor.Labels(["species"], np.array([[1], [6]]))

block_H = metatensor.TensorBlock(
    # atom 0: values [1.0, 2.0], atom 2: values [3.0, 4.0]
    values=np.array([[1.0, 2.0], [3.0, 4.0]]),
    samples=metatensor.Labels(["atom"], np.array([[0], [2]])),
    components=[],
    properties=metatensor.Labels(["n"], np.array([[0], [1]])),
)

block_C = metatensor.TensorBlock(
    # atom 1: values [5.0, 6.0], atom 2: values [7.0, 8.0]
    values=np.array([[5.0, 6.0], [7.0, 8.0]]),
    samples=metatensor.Labels(["atom"], np.array([[1], [2]])),
    components=[],
    properties=metatensor.Labels(["n"], np.array([[0], [1]])),
)

tensor = metatensor.TensorMap(keys, [block_H, block_C])
print(tensor)

# %%
#
# Merge with default fill (zero)
# -------------------------------
#
# Merging along properties with the default ``fill_value=0.0`` fills missing
# entries with zero. Atom 0 has no data for species=6, so those columns are 0:

merged_zero = tensor.keys_to_properties("species")
print(merged_zero.block().values)
# atom 0: [1.0, 2.0, 0.0, 0.0]  -- species=6 columns are zero
# atom 1: [0.0, 0.0, 5.0, 6.0]  -- species=1 columns are zero
# atom 2: [3.0, 4.0, 7.0, 8.0]  -- present in both blocks

# %%
#
# The zeros for atoms 0 and 1 look the same as a genuine zero value. If the
# data could legitimately be zero, there is no way to tell the difference.

# %%
#
# Merge with NaN fill
# --------------------
#
# Setting ``fill_value=float("nan")`` marks missing entries with NaN instead:

merged_nan = tensor.keys_to_properties("species", fill_value=float("nan"))
print(merged_nan.block().values)
# atom 0: [1.0, 2.0, nan, nan]
# atom 1: [nan, nan, 5.0, 6.0]
# atom 2: [3.0, 4.0, 7.0, 8.0]

# %%
#
# Build a missing-data mask
# --------------------------
#
# With NaN fill, ``np.isnan`` identifies exactly which entries were missing:

values = merged_nan.block().values
missing = np.isnan(values)
print("Missing-data mask:")
print(missing)
# [[False False  True  True]
#  [ True  True False False]
#  [False False False False]]

# %%
#
# This is useful for downstream code that needs to handle missing data
# explicitly, for example by masking losses during training.
#
# Note that the fill_value also applies to gradient blocks: if blocks have
# gradients, the gradient arrays for missing entries will also be filled with
# the specified value (e.g. NaN). This ensures consistent missing-data
# semantics across both values and gradients.

# %%
#
# The same ``fill_value`` parameter is available on
# :py:meth:`TensorMap.keys_to_samples`. In this example, all samples are
# disjoint across species, so no entries are missing and no NaN appears:

merged_samples = tensor.keys_to_samples("species", fill_value=float("nan"))
print(merged_samples.block().values)
# [[1. 2.]
#  [5. 6.]
#  [3. 4.]
#  [7. 8.]]
