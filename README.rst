Metatensor
==========

|test| |docs|

Metatensor is a specialized data storage format for all your atomistic machine
learning needs, and more. Think numpy ``ndarray`` or pytorch ``Tensor`` equipped
with extra metadata for atomic — and other particles — systems. The core of this
library is written in Rust and we provide API for C, C++, and Python.

The main class of metatensor is the ``TensorMap`` data structure, defining a
custom block-sparse data format. If you are using metatensor from Python, we
additionally provide a collection of mathematical, logical and other utility
operations to make working with TensorMaps more convenient.

Documentation
-------------

For details, tutorials, and examples, please have a look at our `documentation`_.

Contributors
------------

Thanks goes to all people that make metatensor possible:

.. image:: https://contrib.rocks/image?repo=lab-cosmo/metatensor
   :target: https://github.com/lab-cosmo/metatensor/graphs/contributors

We always welcome new contributors. If you want to help us take a look at
our `contribution guidelines`_ and afterwards you may start with an open issue
marked as `good first issue`_.

.. _`documentation`: https://lab-cosmo.github.io/metatensor/latest/
.. _`contribution guidelines`: CONTRIBUTING.rst
.. _`good first issue`: https://github.com/lab-cosmo/metatensor/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22

.. |test| image:: https://img.shields.io/github/checks-status/lab-cosmo/metatensor/master
   :alt: Github Actions tests status
   :target: https://github.com/lab-cosmo/metatensor/actions?query=branch%3Amaster

.. |docs| image:: https://img.shields.io/badge/documentation-latest-sucess
   :alt: Documentation
   :target: https://lab-cosmo.github.io/metatensor/latest/
