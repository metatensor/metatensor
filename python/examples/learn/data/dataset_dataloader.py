"""
.. _learn-tutorial-dataset-dataloader:

Building a Dataset and DataLoader
=================================

.. py:currentmodule:: metatensor.torch.learn.data

This tutorial shows how to define torch-based Dataset and Dataloader objects
following metatensor's interface.

In metatensor this is based on `PyTorch`_ and `TorchScript`_, so make sure you
are familiar with both before reading this tutorial!

.. _PyTorch: https://pytorch.org/
.. _TorchScript: https://pytorch.org/docs/stable/jit.html
"""

# %%
#
# Let's start by importing things we'll need:

import os
import time

import torch

from metatensor.learn.data import DataLoader, Dataset, group
from metatensor.torch import Labels, TensorBlock, TensorMap


# %%
#
# Now let's define a simple dummy dataset with two data fields,
# named 'x' and 'y'. Every data field passed to `Dataset` must be
# passed as a list of objects corresponding to a unique sample.
#
# Let's define our x data as a list of random 3x3 tensors, and
# our y data as a list of integers that enumerate the samples.

n_samples = 5
x_data = [torch.randn(3, 3) for _ in range(n_samples)]
y_data = [i for i in range(n_samples)]

# %%
#
# In-memory dataset
# -----------------
#
# We are ready to build a dataset. The simplest use case is when all
# data is in memory. In this case, we can pass the data directly to the
# `Dataset` constructor as keywrod arguments, named and ordered according
# to how we want the data to be returned when we access samples in the
# dataset.

dset_mem = Dataset(x=x_data, y=y_data)

# %%
#
# We can now access samples in the dataset. The returned object is a named
# tuple with named fields corresponding to the ones passed to the constructor
# in Dataset initialization. In this case, this is 'x' and 'y'.

print(dset_mem[0])

# %%
#
# Any number of named data fields can be passed to the Dataset constructor, as
# long as they are all unqiuely named, and are all lists of the same length.
# The elements of each list can be any object, as long as it is the same for
# all samples in the respective field.
#
# For example, here we are creating a dataset of torch tensors ('x'), integers
# ('y'), and strings ('z').

dset_mem_1 = Dataset(x=x_data, y=y_data, z=["a", "b", "c", "d", "e"])
print(dset_mem_1[0])
print("Sample 4, z-field:", dset_mem_1[4].z)

# %%
#
# Mixed in-memory / on-disk dataset
# ---------------------------------
#
# Now suppose we have a large dataset, where the x data is too large to fit in
# memory. In this case, we might want to lazily load data when training a model
# with minibatches.
#
# Let's save the x data to disk to simulate this use case.

# Create a directory to save the dummy x data to disk
os.makedirs("x_data", exist_ok=True)

# Important! Save each sample as a separate file
for i, x in enumerate(x_data):
    torch.save(x, f"x_data/x_{i}.pt")

# %%
#
# In order for the x data to be lazly loaded, we need to pass to the Dataset a
# `transform` function that loads into memory the data from disk. This can be
# any callable of arbitrary complexity, but must take as a single argument a
# unique sample ID that can discern and load the data for different samples.


# Define a `transform` function that loads the x data from disk
# and returns a tensor in memory
def transform_x(sample_id):
    """Loads the x data for the sample indexed by `sample_id` from disk and
    returns the object in memory"""
    return torch.load(f"x_data/x_{sample_id}.pt")


print("transform_x fxn called with sample ID 0:", transform_x(0))

# %%
#
# Now when we define a dataset, the 'x' data field can be passed as a callable.

dset_mixed = Dataset(x=transform_x, y=y_data)
print(dset_mixed[3])

# %%
#
# On-disk dataset
# ---------------------------
#
# Finally, suppose we have a large dataset, where both the x and y data are too
# large to fit in memory. In this case, we might want to lazily load all data
# when training a model with minibatches.
#
# Let's now save the y data to disk as well to simulate this use case.

# Create a directory to save the dummy y data to disk
os.makedirs("y_data", exist_ok=True)

# Important! Save each sample as a separate file
for i, y in enumerate(y_data):
    torch.save(y, f"y_data/y_{i}.pt")


# Define a `transform` function that loads the y data from disk
# and returns the integer in memory
def transform_y(sample_id):
    """Loads the y data for the sample indexed by `sample_id` from disk and
    returns the object in memory"""
    return torch.load(f"y_data/y_{sample_id}.pt")


print("transform_y fxn called with sample ID 0:", transform_y(0))

# %%
#
# Now when we define a dataset, as both x and y data are to be lazily loaded, we
# need to indicate how many samples are in the dataset with the `size` argument.
#
# Internally, the Dataset class infers the unique sample IDs as a continuous
# integer sequence starting from 0 to `size` - 1 (inclusive). In this case,
# sample IDs are therefore [0, 1, 2, 3, 4]. These IDs are used to lazily load
# the data upon access.
dset_disk = Dataset(x=transform_x, y=transform_y, size=n_samples)
print(dset_disk[2])

# %%
#
# Building a Dataloader
# ---------------------
#
# Now let's see how we can use the Dataset class to build a DataLoader.
#
# The DataLoader class is a wrapper around the PyTorch DataLoader class, and
# as such can be initialized with just a Dataset object. This will inherit all
# of the default arguments from the PyTorch DataLoader class.

dloader_mem = DataLoader(dset_mem)

# %%
#
# We can now iterate over the DataLoader to access batches of samples from the
# dataset. With no arguments passed, the default batch size is 1 and the samples
# are not shuffled.

for batch in dloader_mem:
    print(batch.y)

# %%
#
# Alternatively, the data fields can be unpacked into separate variables in the
# for loop.

for batch in dloader_mem:
    print(batch.y)

# %%
#
# We can also pass arguments to the DataLoader constructor to change the batch
# size and shuffling of the samples.
dloader_mem = DataLoader(dset_mem, batch_size=2, shuffle=True)

for batch in dloader_mem:
    print(batch.y)

# %%
#
# Let's compare the time it takes to iterate over batches in each of the
# datasets. We will only access the data, over 1000 'epochs'.

dloader_mem = DataLoader(dset_mem)
dloader_mixed = DataLoader(dset_mixed)
dloader_disk = DataLoader(dset_disk)

n_epochs = 1000
print(f"Dataset    |    Time for {n_epochs} epochs (s)")
print("-------------------------------------------------")

t0 = time.time()
for _ in range(n_epochs):
    for _batch in dloader_mem:
        pass
t1 = time.time()
print("In-memory  |   ", t1 - t0)

t0 = time.time()
for _ in range(n_epochs):
    for _batch in dloader_mixed:
        pass

t1 = time.time()
print("Mixed      |   ", t1 - t0)

t0 = time.time()
for _ in range(n_epochs):
    for _batch in dloader_disk:
        pass

t1 = time.time()
print("On-disk    |   ", t1 - t0)

# %%
#
# As expected, the in-memory dataset is the fastest, followed by the mixed
# dataset, and finally the on-disk dataset.

# %%
#
# DataLoaders for cross-validation
# --------------------------------
#
# One can use the torch-native random_split() function to split a
# Dataset into train, validation, and test subsets for cross-validation
# purposes. DataLoaders can then be constructed for each subset.


# Perform a random train/val/test split of the Dataset,
# in the relative proportions 60% : 20% : 20%
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dset_mem, [0.6, 0.2, 0.2]
)

# Construct DataLoaders for each subset
train_dataloader = DataLoader(train_dataset)
val_dataloader = DataLoader(val_dataset)
test_dataloader = DataLoader(test_dataset)

# As the Dataset was initialized with 5 samples, the split should be 3:1:1
print(f"Dataset size: {len(dset_disk)}")
print(f"Training set size: {len(train_dataloader)}")
print(f"Validation set size: {len(val_dataloader)}")
print(f"Test set size: {len(test_dataloader)}")


# %%
#
# Working with torch Tensor and metatensor TensorMap objects
# ----------------------------------------------------------
#
# As the Dataset and DataLoader classes are metatensor/torch interfacing
# classes, let's explore how they behave when using torch Tensor and
# metatensor TensorMap objects as the data.
#
# For the purposes of this part of the tutorial, we will use a mixed in-memory
# / on-disk dataset. Our dummy data will consist of the following fields:
#
# - 'descriptor': a list of random TensorMap objects representing a dummy
#                 descriptor
# - 'scalar': a list of floats representing a dummy scalar target property
# - 'vector': a list of random torch Tensors of size (1, 3) representing
#             a dummy vectorial target property

# Create a dummy descriptor as a TensorMap
descriptor = [
    TensorMap(
        keys=Labels(
            names=["key1", "key2", "key3"],
            values=torch.tensor(
                [
                    [1, 2, 3],
                ]
            ),
        ),
        blocks=[
            TensorBlock(
                values=torch.randn((1, 3)),
                samples=Labels(names=["sample_id"], values=torch.tensor([[sample_id]])),
                components=[],
                properties=Labels(
                    names=["property1", "property2"],
                    values=torch.tensor([[1, 2], [3, 4], [5, 6]]),
                ),
            )
        ],
    )
    for sample_id in range(n_samples)
]

# Create dummy scalar and vectorial target properties as torch Tensors
scalar = [float(torch.rand(1, 1)) for _ in range(n_samples)]
vector = [torch.rand(1, 3) for _ in range(n_samples)]

# Build the Dataset
dset = Dataset(
    descriptor=descriptor,
    scalar=scalar,
    vector=vector,
)
print(dset[0])

# %%
#
# Collating data into a batch
# ---------------------------
#
# When forming a minibatch, a simple way to collate data is to group it by the
# data field it belongs to.
#
# The collate function from the `collate` module, `group`, does just this. It
# takes a list of Sample objects (i.e. named tuples) returned from mutliple
# calls of the Dataset.__getitem__() method, and groups the data by field.

# Construct the DataLoader using the `group` collate function
batch_size = 2
dloader = DataLoader(dset, collate_fn=group, batch_size=batch_size)

batch = next(iter(dloader))
print(batch)

# %%
#
# Returned from the DataLoader.__iter__() method is a Batch object, which is
# a named tuple with named fields corresponding to the data fields in the
# Dataset. In this case, this is 'descriptor', 'scalar', and 'vector'.
#
# Each field is returned as a tuple of length `batch_size`, where each element
# is a the data field for a single sample in the batch.
for field in batch:
    print(field)
    assert len(field) == batch_size

# %%
#
# As is typically customary when working with torch Tensors, we can vertically
# stack the samples in a minibatch into a single Tensor object. This allows
# passing a single Tensor object to a model, rather than a tuple of Tensor
# objects.
#
# In a similar way, sparse data stored in metatensor TensorMap objects can also
# be vertically stacked, i.e. joined along the samples axis, into a single
# TensorMap object.
#
# The default collate function used by the DataLoader class, `group_and_join`,
# vstacks / joins along the samples axis data fields that correspond to torch
# Tensor and metatensor TensorMap objects, respectively. For all other data
# types, the data is just grouped by field.

# Construct the dataloader using the default collate function
batch_size = 2
dloader = DataLoader(
    dset, batch_size=batch_size
)  # by default uses `collate_fn=group_and_join`

# %%
#
# We can look at a single Batch object (i.e. a named tuple, returned by the
# DataLoader.__iter__() method) to see this in action.

batch = next(iter(dloader))

# TensorMaps for each sample in the batch joined along the samples axis
# into a single TensorMap
print(batch.descriptor)

# `scalar` data are float objects, so are just grouped and returned in a tuple
print(batch.scalar)
assert len(batch.scalar) == batch_size

# `vector` data are torch Tensors, so are vertically stacked into a single
# Tensor
print(batch.vector)


# %%
#
# One can also define a custom collate function to use when constructing a
# DataLoader. This can be any callable that accepts a list of Sample objects
# (i.e. named tuples) returned from the Dataset.__getitem__() method.

# %%
#
# Advanced functionality: IndexedDataset
# --------------------------------------
#
# What if we wanted to explicitly define the sample IDs used to store and access
# samples in the dataset?
#
# See the next tutorial, :ref:`learn-tutorial-indexed-dataset-dataloader`,
# for more details.
