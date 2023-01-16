Equistore: data storage for atomistic machine learning
======================================================

Equistore is a specialized data storage format suited to all your atomistic
machine learning needs and more. You can think of it like ``numpy.ndarray`` or
``torch.Tensor``, but carrying extra metadata together with the data.

This metadata can be about the nature of the **objects** being described, about
**how** this object is being described, about **symmetry** properties of the
data (this is especially relevant for equivariant machine learning), different
**sparsity** linked to one-hot encoding of species or **components** of
gradients of the above with respect to various parameters.

For example, the object being described could be "one atom in a structure", or
"a pair of atoms", while the how could be "using SOAP power spectrum features"
or "Hamiltonian matrix elements".

Equistore main concern is about representing and manipulating this metadata,
while using other well established library handle the data itself. We currently
support using arbitrary CPU arrays created by any language (including numpy
arrays), as well as PyTorch Tensor --- including full support for GPU and
automatic differentiation.

.. TODO: the end goal is to create an ecosystem of inter-operable libraries for atomistic ML
.. TODO: equistore does not create data, other libraries do
.. TODO: add a figure

--------------------------------------------------------------------------------

This documentation covers everything you need to know about equistore.
It comprises of the following five broad sections:

- :ref:`userdoc-get-started`: familiarize yourself with equistore and it's
  ecosystem;
- :ref:`userdoc-tutorials`: step-by-step tutorials addressing key problems and
  use-cases for equistore;
- :ref:`userdoc-references`: technical description of all the functionalities
  provided by equistore;
- :ref:`userdoc-explanations`: high-level explanation of more advanced
  functionalities;
- :ref:`devdoc`: how to contribute to the code or the documentation of equistore.

.. toctree::
   :hidden:

   get-started/index
   tutorials/index
   reference/index
   explanations/index
   devdoc/index
