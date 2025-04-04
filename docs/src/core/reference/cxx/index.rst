.. _cxx-api-core:

C++ API reference
=================

.. note::

  This is the documentation for ``metatensor-core`` version
  |metatensor-core-version|. For other versions, look in the following pages:

  .. version-list::
    :tag-prefix: metatensor-core-v
    :url-suffix: core/reference/cxx/index.html

    .. version:: 0.1.12
    .. version:: 0.1.11
    .. version:: 0.1.10
    .. version:: 0.1.9
    .. version:: 0.1.7
    .. version:: 0.1.6
    .. version:: 0.1.5
    .. version:: 0.1.4
    .. version:: 0.1.3

Metatensor offers a C++ API, built on top of the :ref:`C API <c-api-core>`.
You can the provided classes and functions in your own code by :ref:`installing
the corresponding shared library and header <install-c>`, and then including
``metatensor.hpp`` and linking with ``-lmetatensor``. Alternatively, we provide
a cmake package config file, allowing you to do use metatensor like this (after
installation):

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.16)
    project(my-project CXX)

    find_package(metatensor)

    # add executables/libraries
    add_executable(my-exe exe-source.cpp)
    add_library(my-lib lib-source.cpp)

    # Link to metatensor, this makes the header accessible and link to the right
    # libraries.
    #
    # The `metatensor` target will be an alias for `metatensor::shared`
    # or `metatensor::static` depending how you've installed the code.
    target_link_libraries(my-exe metatensor)
    target_link_libraries(my-lib metatensor)

    # alternatively, you can explicitly use the static or shared build of
    # metatensor. Unless you have a very specific need for a static build, we
    # recommend using the shared version of metatensor: this will allow to pass
    # data from your code to any other code using metatensor.

    # target_link_libraries(my-exe metatensor::shared)
    # target_link_libraries(my-exe metatensor::static)

The functions and types provided in ``metatensor.hpp`` can be grouped as follow:

.. toctree::
    :maxdepth: 1

    tensor
    labels
    block
    data
    misc
