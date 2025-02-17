.. _python-api-core:

Python API reference
====================

.. note::

  This is the documentation for ``metatensor-core`` version
  |metatensor-core-version|. For other versions, look in the following pages:

  .. version-list::
    :tag-prefix: metatensor-core-v
    :url-suffix: core/reference/python/index.html

    .. version:: 0.1.12
    .. version:: 0.1.11
    .. version:: 0.1.10
    .. version:: 0.1.9
    .. version:: 0.1.7
    .. version:: 0.1.6
    .. version:: 0.1.5
    .. version:: 0.1.4
    .. version:: 0.1.3


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

Some modules are part of the advanced API that most users should not need to
interact with:

.. toctree::
    :maxdepth: 1

    data
    misc

The Python interface also provides multiple operations, including mathematical,
logical, and other utility functions that can applied on these core objects.
These are part of a separate python package (``metatensor-operations``,
installed by default when installing ``metatensor``); and documented
:ref:`in the corresponding section <python-api-operations>`.
