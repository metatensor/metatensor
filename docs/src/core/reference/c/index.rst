.. _c-api-core:

C API reference
===============

.. note::

  This is the documentation for ``metatensor-core`` version
  |metatensor-core-version|. For other versions, look in the following pages:

  .. version-list::
    :tag-prefix: metatensor-core-v
    :url-suffix: core/reference/c/index.html

    .. version:: 0.1.17
    .. version:: 0.1.16
    .. version:: 0.1.15
    .. version:: 0.1.14
    .. version:: 0.1.13
    .. version:: 0.1.12
    .. version:: 0.1.11
    .. version:: 0.1.10
    .. version:: 0.1.9
    .. version:: 0.1.7
    .. version:: 0.1.6
    .. version:: 0.1.5
    .. version:: 0.1.4
    .. version:: 0.1.3

``metatensor`` offers a C API that can be called from any language able to call
C functions (in particular, this includes Python, Fortran with ``iso_c_env``,
C++, and most languages used nowadays). Convenient wrappers over the C API are
also provided for :ref:`Python <python-api-core>` and :ref:`C++ <cxx-api-core>`
users.

The C API is implemented in Rust. You can use these functions in your own code
by :ref:`installing the corresponding shared library and header <install-c>`,
and then including ``metatensor.h`` while linking with ``-lmetatensor``.
Alternatively, we provide a ``cmake`` configuration file, allowing you to use
``metatensor`` like this (after installation):

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.22)
    project(my-project C)

    find_package(metatensor)

    # add executables/libraries
    add_executable(my-exe exe-source.c)
    add_library(my-lib lib-source.c)

    # Link to metatensor, this makes the header accessible and links to the
    # right libraries.
    #
    # The `metatensor` target will be an alias for `metatensor::shared`
    # or `metatensor::static` depending how you installed the code.
    target_link_libraries(my-exe metatensor)
    target_link_libraries(my-lib metatensor)

    # alternatively, you can explicitly use the static or shared build of
    # metatensor. Unless you have a very specific need for a static build, we
    # recommend using the shared version of metatensor: this will allow you to
    # pass data from your code to other codes which use the same metatensor
    # shared library.

    # target_link_libraries(my-exe metatensor::shared)
    # target_link_libraries(my-exe metatensor::static)

The functions and types provided in ``metatensor.h`` can be grouped in five
main groups:

.. toctree::
    :maxdepth: 1

    tensor
    labels
    block
    data
    misc
