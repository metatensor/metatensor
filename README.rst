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

For details, tutorials, and examples, please have a look at our `documentation`_.

.. _`documentation`: https://lab-cosmo.github.io/equistore/latest/

.. |test| image:: https://github.com/lab-cosmo/equistore/actions/workflows/tests.yml/badge.svg
   :alt: Github Actions Tests Job Status
   :target: https://github.com/lab-cosmo/equistore/actions/workflows/tests.yml

.. |docs| image:: https://img.shields.io/badge/documentation-latest-sucess
   :alt: Documentation
   :target: https://lab-cosmo.github.io/equistore/latest/
