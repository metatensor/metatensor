Welcome to Equistore
=======================
Equistore is a specialized data storage format suited to all your atomistic simulation needs and more. 
Ever worked with large arrays and lost track of what the data represents or how it is stored? Then Equistore is for you! Read on to find out more about it and how you can use it for your own projects. 
When working with large amounts of data, especially relating to atomistic simulations, one often needs access to the metadata such as the nature of the atomic scale objects being represented, various components seprated by symmetry, and .. to name a few. This metadata is implicit when storing this data as an array and it becomes increasingly painstaking to locate entries in the data corresponding to a specific selection of metadata (for example, imagine locating the gradients of the (nlm) component of the representation of atom *i* in structure *A* with respect to another atom *j*) with the size of the data (or the atomic entity). Equistore comes to the rescue by explicitly storing the data in a format governed by the metadata thereby making the data more understable and accessible  

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview/index
   getting-started
   tutorials/index
   how-to/index
   reference/index
