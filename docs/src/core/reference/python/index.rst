.. _python-api-reference:

Python API reference
====================

.. py:currentmodule:: metatensor

Most users will find the Python interface to ``metatensor`` to be the most
convenient to use. This interface is built on top of the C API, and can be
:ref:`installed independently <install-python>`. The functions and classes
provided in ``metatensor`` can be grouped as follows:

- the three core classes: :py:class:`TensorMap`, :py:class:`TensorBlock`,
  and :py:class:`Labels`;
- :ref:`IO functions <python-api-io>` to save and load :py:class:`TensorMap`;
- advanced functionalities like the :ref:`array format <python-api-array>` for
  data storage

The Python interface also provides multiple operations, including mathematical,
logical, and other utility functions that can applied on these core objects.
These are documented in the next :ref:`documentation section
<python-api-operations>`.

.. toctree::
    :maxdepth: 1

    tensor
    block
    labels
    io
    data
    misc
