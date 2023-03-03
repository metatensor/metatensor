Equistore
=========

|test| |docs|

Equistore is a specialized data storage format suited to all your atomistic
machine learning needs and more. Think of an NumPy "ndarray" or a pytorch "Tensor"
carrying extra metadata for atomistic systems.

The core functionality of equistore is its "TensorMap" data structure.
Along with the format equistore also provides a collection of mathematical, logical
as well as utility operations to make the work with TensorMaps convenient.

A main part of the library is written in Rust and we provide APIs for C/C++ and
Python as well.

Documentation
-------------

For details, tutorials, and examples, please have a look at our `documentation`_.

Contributors
------------

Thanks goes to all people that make equistore possible:

.. image:: https://contrib.rocks/image?repo=lab-cosmo/equistore
   :target: https://github.com/lab-cosmo/equistore/graphs/contributors

We always welcome new contributors. If you want to help us take a look at
our `contribution guidelines`_ and afterwards you may start with an open issue
marked as `good first issue`_.

.. _`documentation`: https://lab-cosmo.github.io/equistore/latest/
.. _`contribution guidelines`: CONTRIBUTING.rst
.. _`good first issue`: https://github.com/lab-cosmo/equistore/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22

.. |test| image:: https://github.com/lab-cosmo/equistore/actions/workflows/tests.yml/badge.svg
   :alt: Github Actions Tests Job Status
   :target: https://github.com/lab-cosmo/equistore/actions/workflows/tests.yml

.. |docs| image:: https://img.shields.io/badge/documentation-latest-sucess
   :alt: Documentation
   :target: https://lab-cosmo.github.io/equistore/latest/
