"""
.. _userdoc-tutorials-first-tensormap:

Getting your first Tensormap
============================
"""

# %%
#
# We will start by importing all the required packages: the classic numpy;
# chemfile to load data, and of course equistore.

import numpy as np
from chemfiles import Trajectory
import equistore
import ase
from ase.io import read
from itertools import product
from equistore import TensorBlock, TensorMap,Labels
#Load the frames of the dataset
frames=[]
#Load the dataset

# frames=[]
# with Trajectory('dataset.xyz') as dataset:
#     frames = [f for f in dataset]

frames = read('dataset.xyz', ':10')
print(len(frames))

# %%
#
# Equistore
# ---------

# In this tutorial, we are going to use a new storage format, Equistore,
# https://github.com/lab_cosmo/equistore.

# Creating your first TensorMap
# --------------------------------
# Let us start with the example of storing bond lengths as a TensorMap. We can think of categorizing the data 
# based on the chemical nature (atomic species) of the two atoms involved in the pair. As not all species might
# be present in all the frames, this choice of storage format will enable us to exploit the sparsity. 

# We start by identifying the set of species present in the system.  

species = np.unique(np.hstack([np.unique(f.numbers) for f in frames]))

#Creating all possible species pairs -these will form the keys of our TensorMap
species_pairs = []
for species1 in species: 
	for species2 in species: 
		species_pairs.append((species1, species2)) # or one could have used itertools.product to get the same result, species_pairs = list(product(species, species))


#For each species pairs, find the relevant samples, i.e. the list of all frames and index of atoms in the frame
#that correspond to the species pair in block_samples
block_samples = [] 
for (a1,a2) in  species_pairs:
    frame_samples = []
    for idx_frame, f in enumerate(frames):
        #create tuples of the form (idx_frame, idx_i, idx_j)
        #where idx_i is the index of atoms in the frame such that they have species =a1
        #and idx_j is the index of atoms in the frame such that they have species =a2
        idx_i, idx_j = np.where(f.numbers==a1)[0], np.where(f.numbers==a2)[0]
        frame_samples.append(list(product([idx_frame], idx_i, idx_j)))

    block_samples.append(np.vstack(frame_samples) )

#block_samples will have as many items as in the list of species_pairs

sample_labels = Labels(("structure", "atom_i", "atom_j"), np.asarray(block_samples[0], dtype=np.int32))
# Equistore uses Labels that describe or enumerate each column of the values being considered. For example
# in the code snippet above, we used  labels to specify that the array of sample indices has three columns
# the first column always holds the structure index, whereas the two following columns have info about the atoms

# Labels((name1, name2, name3), [(value1, value2, value3),
#                                (value1, value2, value3),
#                                (value1, value2, value3)])

# For this particular case, each row describes the corresponding row of  "TensorMap.values" that are called samples

#Now we need to find the corresponding values of the bond length for each sample

block_values = []
for (a1,a2) in  species_pairs:
    frame_values = []
    for idx_frame, f in enumerate(frames):
        idx_i, idx_j = np.where(f.numbers==a1)[0], np.where(f.numbers==a2)[0]
        frame_values.append([f.get_all_distances()[i,j] for i in idx_i for j in idx_j])
    block_values.append(np.vstack(frame_values))

#We could have easily merged this operation with the loops above but for clarity we are repeating them here. We use ASE's
#get_all_distances() function to calculate the bond lengths

block_components = [Labels(['spherical_m'], np.asarray([[0]],dtype=np.int32))]
#TODO
#The **components** of the **TensorBlock**, 
# correspond to the equivariant behaviour of the features calculated, with the
# number of **components** = (2 x *lambda* + 1) where lambda tags the behaviour
# under the irreducible SO(3) group action.
block_properties = Labels(['Angstrom'], np.asarray([(0,)],dtype=np.int32))
#TODO

# We have collected all the necessary ingredients to create our first TensorMap. Since a TensorMap is a container
# that holds blocks of data - namely TensorBlocks, let us transform our data to TensorBlock format
blocks=[]
for block_idx, samples in enumerate(block_samples):
    blocks.append(TensorBlock( values = np.hstack(block_values[block_idx]).reshape(-1,1,1),
                               samples = Labels(["structure", "atom_i", "atom_j"], np.asarray(samples, dtype=np.int32)),
                               components = block_components,
                               properties = block_properties
                              )

                 )

# A TensorBlock is the fundamental constituent of a TensorMap. Each Tensorblock is associated with "values" or data array with n-dimensions (here 3 dimensions), each
# identified by a Label. The first dimension refers to the *samples*.

# The last dimension of the n-dimensional array is the one indexing the "properties" or features
# of what we are describing in the TensorBlock.  These also usually correspond to
# all the entries in the basis or :math: `<q|` when the object being represented
# in mathematically expressed as :math: `<q|f>`.
# For the given example, the property dimension is a dummy variable since we are just storing one number corresponding to the bondlength(A).
# But for instance, we could have chosen to project these values on a radial basis <n|r_{ij}>, then
# the properties dimension would correspond to the radial channels or *n* going from 0 up to :math:`n_max`.

# All intermediate dimensions of the array are referred to as *components* and
# are used to describe vectorial or tensorial components of the data.

# A block can also contain gradients of the values with respect to a variety of
# parameters. More about this can be found in the **gradients** section.


bond_lengths = TensorMap(Labels(("species_1", "species_2"), np.asarray(species_pairs, np.int32)), 
                         blocks)

# Here we instantiated a TensorMap by specifying the constituent blocks and their labels (this special class 
# of labels for blocks are also called "keys") 


# Using TensorMaps with Models 
# -----------
# To motivate why this way of storage in Equistore is helpful,let's consider a model built on top of these bondlengths 

# Storing our targets as TensorMaps 
# Just like we did for the bond-lengths, we can create a TensorMap for energies of the structures
energies = np.array([f.info["energy"] for f in frames])

energy_tmap = TensorMap(Labels(["dummy"], np.asarray([(0,)], np.int32)), 
           
          [TensorBlock(values = energies.reshape(-1,1,1) ,
            samples = Labels(["structure"], np.asarray(list(range(len(frames))), dtype=np.int32).reshape(-1,1)),
            components = block_components,
            properties = Labels(['eV'], np.asarray([(0,)],dtype=np.int32))
           )]
              )

#we created a dummy index to address our block of the energy_tmap that just has one tensorblock. 

forces = np.array([f.get_forces() for f in frames])

for idx_frame, f in enumerate(frames): 
    values = f.get_forces()
    samples = Labels(["structure", "center"], np.asarray(list(product([idx_frame],range(len(f)))), dtype=np.int32))
    components = [Labels(["component"], np.asarray([[0,1,2]], np.int32).reshape(-1,1))]
                #cartesian force direction components x=0, y=1, z=2
    properties =  Labels(["eV/A"], np.asarray([[0]])) 

# Accessing different Blocks on the Tensormap
# -------------------------------------------

# There are three main ways to access blocks on the TensorMap, by specifying an index corresponding to the absolute position of the block in the TensorMap, or by specifying the values of one or multiple keys of the TensorMap.

# The first tensorblock can be accessed using
# TensorMap.block(0)

# %%
#
# The second method involves calling the key of the TensorBlock directly using
# the tuple (*spherical_harmonics_l*, *species_center*)
#
# The tensorblock corresponding to key () can be accessed using

# TensorMap.key(key)

# %%
#
# Simple operations on TensorMaps
# -------------------------------
# 
# 1. Reshaping Blocks 
# 2. Reindexing Blocks 
# 3. Restructuring Blocks 
# 
# Keys to properties 
# keys to samples
# components to properties 
# 
# Creating your own TensorBlocks and TensorMap
# --------------------------------------------
# 
# In principle once you have defined blocks and keys, a TensorMap is simply
# obtained by collecting all the blocks into a common container. So how do we
# get these blocks? We need to define **Labels** for each dimension of the
# block values


# list_of_blocks = []
# list_of_blocks.append( TensorBlock(block.values = values,
# block.samples = samples, 
# block. properties = properties,
# block.components = components
# )
# )

# tensormap = TensorMap(blocks, keys)

# &&
#
# where values is an n-dim array with the actual data that you began with,
# whereas samples, properties, components are Label objects. 
# (make sure that the Label objetcs have been appropriately defined to follow
# this explanation)::

# samples = Labels( values, names)

# %%
#
# Going from tensormap to a dense array
# -------------------------------------
