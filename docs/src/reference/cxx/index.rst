.. _cxx-api-reference:

C++ API reference
=================

Equistore offers a C++ API, built on top of the :ref:`C API <c-api-reference>`.
You can the provided classes and functions in your own code by :ref:`installing
the corresponding shared library and header <install-c-lib>`, and then including
``equistore.hpp`` and linking with ``-lequistore``. Alternatively, we provide a
cmake package config file, allowing you to do use equistore like this (after
installation):

.. code-block:: cmake

    find_package(equistore)

    # add executables/libraries
    add_executable(MyExecutable my_sources.cxx)
    add_library(MyLibrary my_sources.cxx)

    # Link to equistore, this makes the header accessible
    target_link_libraries(MyExecutable equistore)

    # alternatively, you can explicitly use the static or shared build of equistore
    # target_link_libraries(MyExecutable equistore::static)
    # target_link_libraries(MyExecutable equistore::shared)

The functions and types provided in ``equistore.hpp`` can be grouped as follow:

.. toctree::
    :maxdepth: 1

    tensor
    labels
    block
    data
    misc
