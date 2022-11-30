"""
.. _userdoc-tutorials-first-tensormap:

Getting your first Tensormap
============================
"""

# %%
#
# We will start by importing all the required packages: the classic numpy;
# ase to load the data, and of course equistore.

import numpy as np
from chemfiles import Trajectory
import equistore
import ase
from ase.io import read
from itertools import product
from equistore import TensorBlock, TensorMap,Labels
frames=[]
#Load the dataset

# frames=[]
# with Trajectory('dataset.xyz') as dataset:
#     frames = [f for f in dataset]

frames = read('dataset.xyz', ':10')

# %%
#
# Equistore
# ---------
#
# In this tutorial, we are going to use a new storage format, Equistore,
# https://github.com/lab_cosmo/equistore.
# %%
# Creating your first TensorMap
# --------------------------------
# Let us start with the example of storing bond lengths as a TensorMap. We can think of categorizing the data 
# based on the chemical nature (atomic species) of the two atoms involved in the pair. As not all species might
# be present in all the frames, this choice of storage format will enable us to exploit the sparsity. 
#
# We start by identifying the set of species present in the system.  

species = np.unique(np.hstack([np.unique(f.numbers) for f in frames]))
# %%
#
#Creating all possible species pairs -these will form the keys of our TensorMap
species_pairs = []
for species1 in species: 
	for species2 in species: 
		species_pairs.append((species1, species2)) # or one could have used itertools.product to get the same result, species_pairs = list(product(species, species))

# %%
#
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
# %%
#
#block_samples will have as many items as in the list of species_pairs

sample_labels = Labels(("structure", "atom_i", "atom_j"), np.asarray(block_samples[0], dtype=np.int32))
# %%
#
# Equistore uses Labels that describe or enumerate each column of the values being considered. For example
# in the code snippet above, we used  labels to specify that the array of sample indices has three columns
# the first column always holds the structure index, whereas the two following columns have info about the atoms

# Labels((name1, name2, name3), [(value1, value2, value3),
#                                (value1, value2, value3),
#                                (value1, value2, value3)])

#%%
# For this particular case, each row describes the corresponding row of  "TensorMap.values" that are called samples
# Now we need to find the corresponding values of the bond length for each sample

block_values = []
for (a1,a2) in  species_pairs:
    frame_values = []
    for idx_frame, f in enumerate(frames):
        idx_i, idx_j = np.where(f.numbers==a1)[0], np.where(f.numbers==a2)[0]
        frame_values.append([f.get_all_distances()[i,j] for i in idx_i for j in idx_j])
    block_values.append(np.vstack(frame_values))
# %%
#
# We could have easily merged this operation with the loops above but for clarity we are repeating them here. We use ASE's
# get_all_distances() function to calculate the bond lengths

block_components = [Labels(['spherical_symmetry'], np.asarray([[0]],dtype=np.int32))]
# spherical_symmetry has just one value = 0 to specify that this quantity is a scalar 

# %%
#
# TODO
# The **components** of the **TensorBlock**, 
# correspond to the equivariant behaviour of the features calculated, with the
# number of **components** = (2 x *lambda* + 1) where lambda tags the behaviour
# under the irreducible SO(3) group action.
block_properties = Labels(['Angstrom'], np.asarray([(0,)],dtype=np.int32))
#TODO
# %%
#
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

# %%
#
# A TensorBlock is the fundamental constituent of a TensorMap. Each Tensorblock is associated with "values" or data array with n-dimensions (here 3 dimensions), each
# identified by a Label. The first dimension refers to the *samples*.
#
# The last dimension of the n-dimensional array is the one indexing the "properties" or features
# of what we are describing in the TensorBlock.  These also usually correspond to
# all the entries in the basis or :math: `<q|` when the object being represented
# in mathematically expressed as :math: `<q|f>`.
# For the given example, the property dimension is a dummy variable since we are just storing one number corresponding to the bondlength(A).
# But for instance, we could have chosen to project these values on a radial basis <n|r_{ij}>, then
# the properties dimension would correspond to the radial channels or *n* going from 0 up to :math:`n_max`.
#
# All intermediate dimensions of the array are referred to as *components* and
# are used to describe vectorial or tensorial components of the data.
#
# A block can also contain gradients of the values with respect to a variety of
# parameters. More about this can be found in the **gradients** section.


bond_lengths = TensorMap(Labels(("species_1", "species_2"), np.asarray(species_pairs, np.int32)), 
                         blocks)
# %%
#
# Here we instantiated a TensorMap by specifying the constituent blocks and their labels (this special class 
# of labels for blocks are also called "keys") 

# %%
# Storing potential targets as TensorMaps 
# -------------------------------------------
# Before we use the bond lengths with models to predict the energies of the structure, lets also briefly look at how potential targets such as energies or forces would be stored as Equistore TensorMaps.  
#
# Just like we did for the bond-lengths, we can create a TensorMap for energies of the structures
energies = np.array([f.info["energy"] for f in frames])

energy_tmap = TensorMap(Labels(["dummy"], np.asarray([(0,)], np.int32)),  
          [TensorBlock(values = energies.reshape(-1,1,1) ,
            samples = Labels(["structure"], np.asarray(list(range(len(frames))), dtype=np.int32).reshape(-1,1)),
            components = block_components,
            properties = Labels(['eV'], np.asarray([(0,)],dtype=np.int32))
           )]
		       )
# %%
# we created a dummy index to address our block of the energy_tmap that just has one tensorblock. 
force_values = []
force_samples= []
for idx_frame, f in enumerate(frames): 
    force_values.append(f.get_forces())
    force_samples.append(np.asarray(list(product([idx_frame],range(len(f)))), dtype=np.int32))
    
force_values =  np.vstack(force_values)
force_components= [Labels(["component"], np.asarray([[0,1,2]], np.int32).reshape(-1,1))]
#cartesian force direction components x=0, y=1, z=2    
force_properties = Labels(["eVperA"], np.asarray([[0]]))

force_tmap = TensorMap(Labels(["dummy"], np.asarray([(0,)], np.int32)),     
            [TensorBlock(values= force_values.reshape((force_values.shape)+(1,)),
            samples= Labels(["structure", "center"], np.vstack(force_samples)), 
            components= force_components, 
            properties=force_properties
            )]
                      ) 

# %%
# Summary to building Tensormaps
# ---------------------------------------------
#
# A TensorMap is simply obtained by collecting some Tensorblocks, each addressed by a key value, into a common container. The TensorBlocks contain within them the actual values (some n-dimensional array or tensor) you might be interested in working with, but carry along the Labels specifying what each dimension corresponds to. 

# list_of_blocks = []
# list_of_blocks.append( TensorBlock(block.values = values,
#                                    block.samples = samples, 
#                                    block. properties = properties,
#                                    block.components = components
#                                    ))
# tmap = TensorMap(keys, list_of_blocks) 

# %%
#
# Accessing different Blocks of the Tensormap
#------------------------------------------------
#
# There are multiple ways to access blocks on the TensorMap, either by specifying the index value corresponding to the absolute position of the block in the TensorMap, 
# or by specifying the values of one or multiple keys of the TensorMap.
# For instance, the first tensorblock can be accessed using TensorMap.block(0)
# In the example above, 

energy_tmap.block(0) 

# %%
# just returns the only block in the energy tensormap, whereas 
bond_lengths.block(1) 

# %%
# returns the block corresponding to key = bond_lengths.keys[1] (that happens to be the H-C block, i.e. species_1 = 1 and species_2 = 6) 
# 
# The second method involves specifying the values of the keys of the TensorBlock directly, for instance if we are interested in the bond length block between H and C, we can also get them 
bond_lengths.block(species_1 = 1, species_2 = 6) 

# %%
# If we are just interested in blocks that have the first atom as H irrespective of the species of the second atom, 
bond_lengths.blocks(species_1 = 1) 
# %%
# Notice that we use TensorMap.block**s** as more than one block satisfies the selection criteria. This returns the list of relevant block. If one is interested in identifying the 
# indices of these blocks in the TensorMap, 
bond_lengths.blocks_matching(species_1 = 1)

# %%
# precisely returns the list of indices of all the blocks where species_1 = 1 (namely 0,1,2,3) and one can then use these indices to also identify the corresponding keys
bond_lengths.keys[bond_lengths.blocks_matching(species_1 = 1)]

# %%
#
# Simple operations on TensorMaps
# -------------------------------------------
# 
# 1. Reshaping Blocks 
# 2. Reindexing Blocks 
# 3. Restructuring Blocks 
# 
# Keys to properties 
# keys to samples
# components to properties 
# Going from tensormap to a dense array


# %%
#
# Training your first model using Equistore
# ----------------------------------------------
# To demonstrate the accessibility and flexiblity of Equistore, we are going to use the polynomial features of the bond lengths, with a cutoff based on the 
# atomic number of the species involved, to predict the energy of the system. 

Cutoff = {
    1 : 2,
    6 : 3,
    7 : 4,
    8 : 4
}

# %%
#
# As Equistore indexes features based on their metadata, it facillitates the implementation of customizable feature engineering for different feature subsets.
# In the following code block, we will build the polynomial features of the bond lengths, with its corresponding cutoff. For instance,
# the bond length of C-H will have its cutoff at :math:`3 + 2 = 5`, meaning that we will take polynomial features of C-H up to degree 5.

training_features = []

for (i, j) in bond_lengths.keys:
    block = bond_lengths.block(species_1 = i, species_2 = j)
    Polynomial_Cutoff = Cutoff[i] + Cutoff[j]
    all_structures = np.unique(block.samples["structure"])
    individual_features = []
    for structure_i in all_structures:
        polynomial_features = []
        atoms_i = block.samples["structure"] == structure_i
        for power in range(Polynomial_Cutoff):
            feature = np.sum(np.power(block.values[atoms_i,:], power + 1), axis = 0)
            polynomial_features.append(feature)
        individual_features.append(np.array(polynomial_features).squeeze())
    structure_feature = np.vstack(individual_features)
    training_features.append(structure_feature)
    
training_features = np.hstack(training_features)
    
# %%
#
# Using these features, we can now build our model to predict the energy of the system. For the sake of simplicity, we are going to use Sklearn's implementation 
# for Linear Regression.

from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(training_features, energies)
print ("The R2 score for our model is {}".format(model.score(training_features, energies)))
