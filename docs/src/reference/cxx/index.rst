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

    # Link to equistore, this makes the header accessible and link to the right
    # libraries.
    #
    # The `equistore` target will be an alias for `equistore::shared`
    # or `equistore::static` depending how you've installed the code.
    target_link_libraries(MyExecutable equistore)
    target_link_libraries(MyLibrary equistore)

    # alternatively, you can explicitly use the static or shared build of
    # equistore. Unless you have a very specific need for a static build, we
    # recommend using the shared version of equistore: this will allow to pass
    # data from your code to any other code using equistore.

    # target_link_libraries(MyExecutable equistore::shared)
    # target_link_libraries(MyExecutable equistore::static)

The functions and types provided in ``equistore.hpp`` can be grouped as follow:

.. toctree::
    :maxdepth: 1

    tensor
    labels
    block
    data
    misc
