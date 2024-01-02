"""
.. _learn-tutorial-indexed-dataset-dataloader:

Using an IndexedDataset
=======================

.. py:currentmodule:: metatensor.torch.learn.data

This tutorial shows how to define a metatensor/torch IndexedDataset
for samples with specific sample IDs.

In metatensor this is based on `PyTorch`_ and `TorchScript`_, so make sure you
are familiar with both before reading this tutorial!

.. _PyTorch: https://pytorch.org/
.. _TorchScript: https://pytorch.org/docs/stable/jit.html
"""

# %%
#
# Let's start by importing things we'll need:

import os

import torch

from metatensor.learn.data import DataLoader, Dataset, IndexedDataset


# %%
#
# Review of the standard Dataset
# ------------------------------
#
# The previous tutorial, :ref:`learn-tutorial-dataset-dataloader`, showed how to
# define a metatensor/torch Dataset. Each data field is passed as a keyword
# argument to the :class:`~metatensor.torch.learn.data.Dataset` constructor.
#
# We saw that in-memory, on-disk, or mixed in-memory/on-disk datasets can be
# defined. DataLoaders are then defined on top of these Dataset objects.
#
# In every case, however, the each data sample is accessed by a numeric integer
# index, which ranges from 0 to ``len(dataset) - 1``. Let's use a simple example
# to review this.
#
# Again let's define some dummy data as before. Our x data as a list of random
# 3x3 tensors, and our y data as a list of integers that enumerate the samples.
#
# For the purposes of this tutorial, we will only focus on an in-memory dataset,
# though the same principles apply to on-disk and mixed datasets.

n_samples = 5
x_data = [torch.randn(3, 3) for _ in range(n_samples)]
y_data = [i for i in range(n_samples)]

dset = Dataset(x=x_data, y=y_data)

# %%
#
# A sample is accessed by its numeric index. As the len of the lists passed as
# kwargs is 5, both for ``x`` and ``y``, the valid indices are [0, 1, 2, 3, 4].
#
# Let's retrieve the 4th sample (index 3) and print it. The value of the "y"
# data field should be 3.

print(dset[3])

# %%
#
# To reiterate, when defining a DataLoader on top of this, batches of samples
# are accessed by iterating over the dataloader.

# Build DataLoader
dloader = DataLoader(dset, batch_size=2)

# Iterate over batches
for batch in dloader:
    print(batch)

# %%
#
# Alternative iteration over DataLoader
for x, y in dloader:
    print(x, y)

# %%
#
# What if we wanted to access samples by something other than an integer
# index part of a continuous range?
#
# For instance, what if we wanted to access samples by:
#    1. a string ID, or other arbitrary hashable object?
#    2. an integer index that is subset a shuffled and/or
#       subset of a continuous range?
#
# In these cases, we can use an IndexedDataset instead.

# %%
#
# In-memory IndexedDataset with arbitrary sample IDs
# --------------------------------------------------
#
# First let's define a Dataset where the samples are indexed by
# arbitrary unique IDs, such as strings, integers, and tuples.
#
# Suppose the unique IDs for our 5 samples are:

sample_ids = ["cat", 4, ("foo", "bar"), "dog", 0]

# Build an IndexedDataset, specifying the unique sample IDs
idset_a = IndexedDataset(
    x=x_data,
    y=y_data,
    sample_ids=sample_ids,
)

# %%
#
# Now, when we access the dataset, we can access samples by their unique sample
# index using the `get_sample` method. This method takes a single argument, the
# sample ID, and returns the corresponding sample.

print(idset_a.get_sample("dog"))
print(idset_a.get_sample(4))
print(idset_a.get_sample(("foo", "bar")))

# %%
#
# Note that using the ``__get_item__``, i.e. ``idset_a[4]``, will return the
# sample passed to the constructor at numeric index 4. Under the hood, the
# ``get_sample`` method maps the unique sample ID back to the
# internally-used numeric index, and calls ``__get_item__``.
#
# In this case, the sample IDs map to the numeric indices as follows:
#    "cat"          -> 0
#    4              -> 1
#    ("foo", "bar") -> 2
#    "dog"          -> 3
#    0              -> 4
#
# Thus, accessing the unique sample ID "cat" can be done equivalently with both of:

print(idset_a.get_sample("cat"))
print(idset_a[0])

# %%
#
# Note that the named tuple returned in both cases contains the unique
# sample ID as the "sample_id" field, which precedes all other data fields.
# This is in contrast to the standard Dataset, which only returns the passed
# data fields and not the index/ID.
#
# A DataLoader can be constructed on top of an IndexedDataset in the same way
# as a Dataset. Batches are accessed by iterating over the DataLoader, though
# this time the "Batch" named tuple returned by the DataLoader will contain the
# unique sample IDs "sample_ids" (note: plural for the batch, singular for the
# sample) as the first field.

# Build DataLoader
idloader_a = DataLoader(idset_a, batch_size=2)

# Iterate over batches
for batch in idloader_a:
    print(batch)

# %%
#
# Alternative iteration over DataLoader
for ids, x, y in idloader_a:
    print(ids, x, y)

# %%
#
# On-disk IndexedDataset with arbitrary sample IDs
# --------------------------------------------------
#
# When defining an IndexedDataset with data fields on-disk, i.e. to be lazily
# loaded, the sample IDs passed as the ``sample_ids`` kwarg to the constructor
# are used as the arguments to the transform function.
#
# To demonstrate this, as we did in the previous tutorial, let's save the ``x``
# data to disk and build a mixed in-memory/on-disk IndexedDataset.
#
# For instance, the below code will save sone x data for the sample "dog"
# at relative path "x_data_indexed/x_dog.pt".

# Create a directory to save the dummy x data to disk
os.makedirs("x_data_indexed", exist_ok=True)

# Important! Save each sample as a separate file
for i, x in zip(sample_ids, x_data):
    torch.save(x, f"x_data_indexed/x_{i}.pt")

# %%
#
# Define the transform function that loads data from disk. This should take the
# unique sample ID as a single argument, and return the corresponding data in
# memory.


def transform_x(sample_id):
    """Loads the x data for the sample indexed by `sample_id` from disk and
    returns the object in memory"""
    return torch.load(f"x_data_indexed/x_{sample_id}.pt")


# %%
#
# Now when we define an IndexedDataset, the 'x' data field can be passed as a
# callable.

idset_mixed = IndexedDataset(x=transform_x, y=y_data, sample_ids=sample_ids)
print(idset_mixed.get_sample("dog"))
print(idset_mixed.get_sample(("foo", "bar")))


# %%
#
# Using an IndexedDataset: subset integer ranges
# ----------------------------------------------
#
# One could also define an IndexedDataset where the samples indices are
# integers, but form a possibly shuffled and non-continuous) subset of a larger
# continuous range of numeric indices.
#
# For instance, imagine we have a global Dataset of 1000 samples, with indices
# [0, ..., 999], but only want to build a dataset for samples with indices [4,
# 7, 200, 5, 999], in that order. We can pass these indices kwarg ``sample_ids``.

# Build an IndexedDataset, specifying the subset sample IDs in a specific order
sample_ids_b = [4, 7, 200, 5, 999]
idset_b = IndexedDataset(x=x_data, y=y_data, sample_ids=sample_ids_b)

# %%
#
# Now, when we access the dataset, we can access samples by their unique sample
# index using the `get_sample` method. This method takes a single argument, the
# sample ID, and returns the corresponding sample.
#
# Again, the numeric index can be used equivalently to access the sample, and
# again note that the "Sample" named tuple includes the "sample_id' field.

# These return the same sample
print(idset_b.get_sample(5))
print(idset_b[4])

# %%
#
# And finally, the DataLoader behaves as expected:

# Build DataLoader
idloader_b = DataLoader(idset_b, batch_size=2)

# Iterate over batches
for batch in idloader_b:
    print(batch)

# %%
#
# Alternative iteration over DataLoader
for ids, x, y in idloader_b:
    print(ids, x, y)
