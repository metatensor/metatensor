What is equistore
=================

Equistore is a specialized data storage format suited to all your atomistic
simulation needs and more. Equistore provides an accessible and understandable
storage format for the data one comes across in atomistic machine learning.

When working with large amounts of data, especially relating to atomistic
simulations, one often needs access to the metadata such as the nature of
the atomic scale objects being represented, various components seprated by
symmetry, and .. to name a few. This metadata is implicit when storing this
data as an array and it becomes increasingly painstaking to locate entries
in the data corresponding to a specific selection of metadata (for example,
imagine locating the gradients of the (nlm) component of the representation
of atom *i* in structure *A* with respect to another atom *j*) with the size
of the data (or the atomic entity).

Another example arises when using equistore
to compute atom-centered density correlation (ACDC) features, we can divide the
descriptor data into blocks indexed by the chemical nature of the centers,
behavior under symmetry operations (rotational and inversion), and the correlation
order of the representation. Higher order features (in terms of correlations
around the same center or including higher number of centers) can be computed
by combining these blocks, a process that helps highlight their roles in model
performance and tracks the information flow completely.
This data that has been unraveled and stored into different blocks can be
reassembled to a contiguous storage format, if desired, with a step-by-step
control of data recombination.
