Overview
i=======

What is Equistore?
--------

Equistore provides an accessible and understandable storage format for the data one comes across in atomistic machine learning. For instance when using Equistore to compute atom-centered density correlation (ACDC) features, we can divide the descriptor data into blocks indexed by the chemical nature of the centers, behavior under symmetry operations (rotational and inversion), and the correlation order of the representation. Higher order features (in terms of correlations around the same center or including higher number of centers) can be computed by combining these blocks, a process that helps highlight their roles in model performance and tracks the information flow completely.
This data that has been unraveled and stored into different blocks can be reassembled to a contiguous storage format, if desired, with a step-by-step control of data recombination. 

Equistore at a glance
--------
Let's have a brief look at the different kinds of data structures you might encounter when working with Equistore. For the purpose of making this explanation more tangible, let's consider we are working with a dataset of  two water molecules [(O, H, H), (O,H,H)] and hyperparameters for computing the ACDCs to be  (:math:`n_max = 3` and :math:`l_max = 2`).

TensorMap 
######
A TensorMap object can be thought of as a container that holds blocks of data (and all the other following data structures) within it. It is the main object that you will encounter when using this library and can represent any data, for example the spherical expansion coefficients (ACDC of correlation order :math:`\nu = 1`) or the SOAP power spectrum(ACDC of correlation order :math:`\nu = 2` ) could be represented as a TensorMap. TensorMaps contain a list **TensorBlocks**, each addressed by a corresponding **key**. The keys reflect the nature of the atomic scale objects being represented and some crucial attributes to enable such sparsification.
The closest analogy would be that of a *dict* object, that is similarly built on a collection of keys and corresponding values. 
It should be noted that the TensorMap is not just restricted to describe representations or descriptors, but could also very well be used to depict the targets of ML models, such as dipole moments or the effective single particle Hamiltonians. (see the how-to section for more details)

For the example we are considering in this section, the TensorMap 
TensorBlock
#######
A TensorBlock is the fundamental constituent of a TensorMap. Each block is addressed by a key of the TensorMap, so in the example above we would have six blocks for each of the keys. 

Each block in turn is associated with a data array with n-dimensions, each identified by a label. The first dimension refers to the *samples* that are tuples designating the data points that correspond to the key with which the block is associated. 
The TensorBlock associated with the key (*spherical_harmonics_l* = 1, *species_center* = 1) would have entries that contain information about the structure and index of the atoms that (here have species Hydrogen) thus yielding the samples to be [(0,1), (0,2), (1,1), (1,2)]. 

The last dimension of the n-dimensional array indexes the properties or features of what we are describing in the TensorBlock.  These also usually correspond to all the entries in the basis or :math: `<q|` when the object being represented in mathematically expressed as :math: `<q|f>`.
For the given example, the property dimension would correspond to the radial channels or *n* going from 0 up to :math:`n_max`. [(0), (1), (2), (3)]

All intermediate dimensions of the array are referred to as *components* and are used to describe vectorial or tensorial components of the data.  

A block can also contain gradients of the values with respect to a variety of parameters. More about this can be found in the **gradients** section. 


Labels
#######
The set of keys is a equistore.Labels obj    ect, that also keeps track of the names that describe each index in the key.

Each key of the TensorMap would be a tuple of the form (*spherical_harmonics_l*, *species_center*) and be linked to a corresponding TensorBlock. For this example, the list of keys would be [(0,1), (0,8), (1,1), (1,8), (2,1), (2,8)].


Gradients and how we manage them 
-------
Gradient samples - "special" format 

    first sample of gradients is "sample" that refers to the row in block.values that we are taking the gradient of. 
the other samples - what we are taking the gradient with respect to. 
Write what this entails -- block.gradients.sample (i A j) (pair feature i j A k)

Cell gradients  - Sample (i) 
components [[x y z ] [x y z]] (displacement matrix) 

Gradient wrt hypers








