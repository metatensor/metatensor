.. _c-api-reference:

C API reference
===============

``aml_storage`` offers a C API that can be called from any language able to call
C functions (in particular, this includes Python, Fortran with ``iso_c_env``,
C++, and most languages used nowadays). Convenient wrappers of the C API are
also provided for :ref:`Python <python-api-reference>` users.

The C API is implemented in Rust. You can use these functions in your own code
by :ref:`installing the corresponding shared library and header
<install-c-lib>`, and then including ``aml_storage.h`` and linking with
``-laml_storage``. Alternatively, we provide a cmake package config file,
allowing you to do use ``aml_storage`` like this (after installation):

.. code-block:: cmake

    find_package(aml_storage)

    # add executables/libraries
    add_executable(MyExecutable my_sources.c)
    add_library(MyLibrary my_sources.c)

    # Links aml_storage with MyExecutable, this makes the header accessible
    target_link_libraries(MyExecutable aml_storage)

The functions and types provided in ``aml_storage.h`` can be grouped in five
main groups:

.. toctree::
    :maxdepth: 1

    descriptors
    labels
    blocks
    data
    misc
