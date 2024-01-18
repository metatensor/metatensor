.. _python-api-reference:

Python API reference
====================

.. note::

  This is the documentation for ``metatensor-core`` version
  |metatensor-core-version|. For other versions, look in the following pages:

  .. grid::
    :margin: 0 0 0 0

    .. grid-item-card:: Version 0.1.0
      :link: https://lab-cosmo.github.io/metatensor/metatensor-core-v0.1.0/reference/python/index.html
      :link-type: url
      :columns: 12 6 3 3
      :text-align: center
      :class-body: sd-p-2
      :class-title: sd-mb-0

    .. grid-item-card:: Version 0.1.1
      :link: https://lab-cosmo.github.io/metatensor/metatensor-core-v0.1.1/reference/python/index.html
      :link-type: url
      :columns: 12 6 3 3
      :text-align: center
      :class-body: sd-p-2
      :class-title: sd-mb-0


.. py:currentmodule:: metatensor

Most users will find the Python interface to ``metatensor`` to be the most
convenient to use. This interface is built on top of the C API, and can be
:ref:`installed independently <install-python>`. The functions and classes
provided in ``metatensor`` can be grouped as follows:

.. toctree::
    :maxdepth: 1

    tensor
    block
    labels
    io
    data
    misc

The Python interface also provides multiple operations, including mathematical,
logical, and other utility functions that can applied on these core objects.
These are part of a separate python package (``metatensor-operations``,
installed by default when installing ``metatensor``); and documented
:ref:`in the corresponding section <python-api-operations>`.
