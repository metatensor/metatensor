Core classes
============

These pages explain the main classes and core concepts you need to understand
how to use metatensor, without tying it to how a specific library uses
metatensor to store data.

.. grid::

    .. grid-item-card:: 🔬 Core classes overview
        :link: core-classes-overview
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0


        Explore the core types of metatensor: ``TensorMap``, ``TensorBlock`` and
        ``Labels``, and how they are used to store data, metadata and gradients
        of the data together.

    .. grid-item-card:: 💡 Tutorials
        :link: core-tutorials
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0


        Learn how to use the core classes with Python code examples, and improve your
        understanding of data and metadata storage with metatensor.

Metatensor main classes are available from multiple languages, all documented
below.

.. grid::

    .. grid-item-card:: |Python-16x16| Python API reference
        :link: python-api-core
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Read the documentation for all the classes and functions in the
        ``metatensor`` and ``metatensor-core`` Python packages.

        +++
        Documentation for version |metatensor-core-version|

    .. grid-item-card:: |Cxx-16x16| C++ API reference
        :link: cxx-api-core
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Read the documentation for all the classes and functions provided by
        the metatensor C++ API in the ``metatensor.hpp`` header.

        +++
        Documentation for version |metatensor-core-version|

    .. grid-item-card:: |C-16x16| C API reference
        :link: c-api-core
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Read the documentation for all the types and functions provided by
        the metatensor C API in the ``metatensor.h`` header.

        +++
        Documentation for version |metatensor-core-version|

    .. grid-item-card:: |Rust-16x16| Rust API reference
        :link: rust-api-core
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Read the documentation for all the types and functions provided by
        the metatensor Rust API in the ``metatensor`` crate.

        +++
        Documentation for version |metatensor-core-version|


.. toctree::
    :maxdepth: 2
    :hidden:

    overview
    reference/python/index
    reference/cxx/index
    reference/c/index
    reference/rust/index
    ../examples/core/index

.. toctree::
    :maxdepth: 1
    :hidden:

    CHANGELOG.md
