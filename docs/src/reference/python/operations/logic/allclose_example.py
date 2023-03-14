#!/usr/bin/env python
# coding: utf-8

import equistore, numpy as np
from equistore import TensorBlock, TensorMap, Labels, sum_over_samples
from equistore import allclose, allclose_raise, allclose_block, allclose_block_raise

# Create some TensorBlocks

# Create simple block1
block1 = TensorBlock(
    values=np.array([
         [1, 2, 4],
         [3, 5, 6],
     ]),
    samples=Labels(
         ["structure", "center"],
         np.array([
             [0, 0],
             [0, 1],
         ]),
     ),
    components=[],
    properties=Labels(
        ["properties"], np.array([[0], [1], [2]])
     ),
)
keys = Labels(names=["key"], values=np.array([[0]]))

# Recreate block1, but rename properties label 'properties' to 'PROPERTIES'
block2 = TensorBlock(
    values=np.array([
         [1, 2, 4],
         [3, 5, 6],
     ]),
    samples=Labels(
         ["structure", "center"],
         np.array([
             [0, 0],
             [0, 1],
         ]),
     ),
    components=[],
    properties=Labels(
        ["PROPERTIES"], np.array([[0], [1], [2]])
     ),
)

allclose_block(block1, block2)
#Output:
# False
### This fails because allclose_block checks that the properties of two blocks are equal too

# Using allclose_block_raise raises a ValueError when the check fails, and provides additional information
# as to why it fails
allclose_block_raise(block1, block2)
#Output:
# ValueError: Inputs to 'allclose' should have the same properties:
# properties names are not the same or not in the same order.

# Recreate block1, but change first value in the block from 1 to 1.00001
block3 = TensorBlock(
    values=np.array([
         [1.00001, 2, 4],
         [3, 5, 6],
     ]),
    samples=Labels(
         ["structure", "center"],
         np.array([
             [0, 0],
             [0, 1],
         ]),
     ),
    components=[],
    properties=Labels(
        ["properties"], np.array([[0], [1], [2]])
     ),
)

allclose_block(block1, block3)
#Output:
# False
### This fails because the default rtol is 1e-13, and the difference in the first value
### between the two blocks is on the order of 1e-6

allclose_block(block1, block3, rtol=1e-5)
#Output:
# True
### This passes because we have defined the rtol as 1e-6

# Create tensors from blocks, using the same keys
keys = Labels(names=["key"], values=np.array([[0]]))

tensor1 = TensorMap(keys, [block1])
tensor2 = TensorMap(keys, [block2])
tensor3 = TensorMap(keys, [block3])

allclose(tensor1, tensor2)
#Output:
# False

allclose(tensor1, tensor3)
#Output:
# False

allclose(tensor1, tensor3, rtol=1e-5)
#Output:
# True

###These pass/fail for the same reasons as in allclose_block above


# Let's copy the block from tensor3, and create a slightly new tensor with a 
# different value for the keys
block4 = tensor3.block().copy()
keys4 = Labels(names=["key"], values=np.array([[1]]))
tensor4 = TensorMap(keys4, [block4])

allclose(tensor1, tensor4, rtol=1e-5)
#Output:
# False
### This fails, because allclose also checks that the keys of each tensor are equal

# Using allclose_raise will raise a ValueError if the check fails, along with an explanation as to why 
allclose_raise(tensor1, tensor4, rtol=1e-5)
#Output:
# ValueError: Inputs to allclose should have the same key indices.
